"""
DMD + RL (Reinforcement Learning) Model

This module implements the combination of Distribution Matching Distillation (DMD)
with Reinforcement Learning from video reward models.

Loss: loss_gen = dmd_loss + rl_loss_weight * rl_loss

Features:
- RL loss computed from VAE-decoded videos using DifferentiableVideoReward
- Cold start: RL loss only participates after a specified number of training steps
- Reward normalization support for stable training
- Memory-efficient latent window decoding: only a random contiguous window of
  latents is VAE-decoded instead of the full sequence
- Gradient checkpointing on VAE decode and reward model (Qwen2-VL)
- FSDP support for the reward model
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
import torch.distributed as dist
from typing import Optional, Tuple

from model.dmd import DMD


class DMDRL(DMD):
    """
    DMD + RL Model combining distribution matching distillation with
    reinforcement learning from video reward models.
    """

    def __init__(self, args, device):
        super().__init__(args, device)

        # RL hyperparameters
        self.rl_loss_weight = getattr(args, "rl_loss_weight", 1.0)
        self.rl_cold_start_steps = getattr(args, "rl_cold_start_steps", 0)
        self.rl_enabled = False  # Will be enabled after cold start

        # Reward model configuration
        self.rl_reward_checkpoint = getattr(args, "rl_reward_checkpoint", None)
        self.rl_target_height = getattr(args, "rl_target_height", 336)  # Must be multiple of 28
        self.rl_target_width = getattr(args, "rl_target_width", 504)    # Must be multiple of 28
        self.rl_reward_type = getattr(args, "rl_reward_type", "overall")  # VQ, MQ, TA, or overall

        # Memory optimization options
        self.rl_reward_fsdp = getattr(args, "rl_reward_fsdp", False)

        # Latent window: decode only a random contiguous window of latents
        # instead of the full 21-frame sequence. This saves ~4x VAE memory.
        # 0 = decode all latents (no windowing)
        self.rl_latent_window_size = getattr(args, "rl_latent_window_size", 5)

        # Pixel-level frame sampling on top of window decode
        # 0 = no sampling (use all decoded pixel frames)
        self.rl_reward_num_frames = getattr(args, "rl_reward_num_frames", 0)

        # Reward normalization: use inference_config from reward model checkpoint
        self._reward_norm_mean = None
        self._reward_norm_std = None

        # Running EMA for reward monitoring only (not used in loss computation)
        self.rl_reward_ema_mean = 0.0
        self.rl_reward_ema_std = 1.0
        self.rl_reward_ema_decay = getattr(args, "rl_reward_ema_decay", 0.99)
        self._reward_stats_initialized = False

        # Validate target dimensions
        assert self.rl_target_height % 28 == 0, f"rl_target_height must be multiple of 28, got {self.rl_target_height}"
        assert self.rl_target_width % 28 == 0, f"rl_target_width must be multiple of 28, got {self.rl_target_width}"

        # Lazy initialization of reward model (to save memory during cold start)
        self._reward_model = None
        self._reward_model_initialized = False

    def _initialize_reward_model(self):
        """
        Lazily initialize the reward model when RL is first enabled.
        Enables gradient checkpointing and optional FSDP wrapping.
        """
        if self._reward_model_initialized:
            return

        if self.rl_reward_checkpoint is None:
            raise ValueError("rl_reward_checkpoint must be specified for RL training")

        from VideoAlign.inference import DifferentiableVideoReward

        print(f"[DMDRL] Initializing reward model from {self.rl_reward_checkpoint}")
        self._reward_model = DifferentiableVideoReward(
            load_from_pretrained=self.rl_reward_checkpoint,
            device=self.device,
            dtype=self.dtype
        )

        # Freeze reward model parameters
        # Gradients can still flow THROUGH the reward model back to generator
        self._reward_model.inferencer.model.requires_grad_(False)

        # Enable gradient checkpointing on the reward model (Qwen2VL supports this)
        reward_qwen_model = self._reward_model.inferencer.model
        if hasattr(reward_qwen_model, 'gradient_checkpointing_enable'):
            reward_qwen_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(f"[DMDRL] Reward model gradient checkpointing enabled")
        else:
            print(f"[DMDRL] Warning: reward model does not support gradient_checkpointing_enable")

        # Optional FSDP wrapping for the reward model
        if self.rl_reward_fsdp:
            self._fsdp_wrap_reward_model()

        # Load normalization stats from inference_config in the checkpoint
        inference_config = self._reward_model.inferencer.inference_config
        if inference_config is not None:
            self._reward_norm_mean = torch.tensor(
                [inference_config['VQ_mean'], inference_config['MQ_mean'], inference_config['TA_mean']],
                device=self.device, dtype=torch.float32
            )
            self._reward_norm_std = torch.tensor(
                [inference_config['VQ_std'], inference_config['MQ_std'], inference_config['TA_std']],
                device=self.device, dtype=torch.float32
            )
            print(f"[DMDRL] Reward normalization loaded from inference_config: "
                  f"VQ({inference_config['VQ_mean']:.4f}±{inference_config['VQ_std']:.4f}), "
                  f"MQ({inference_config['MQ_mean']:.4f}±{inference_config['MQ_std']:.4f}), "
                  f"TA({inference_config['TA_mean']:.4f}±{inference_config['TA_std']:.4f})")
        else:
            print(f"[DMDRL] Warning: no inference_config found in checkpoint, "
                  f"using raw logits without normalization")

        self._reward_model_initialized = True
        window_info = f"latent window={self.rl_latent_window_size}" if self.rl_latent_window_size > 0 else "full decode"
        print(f"[DMDRL] Reward model initialized and frozen successfully")
        print(f"[DMDRL] VAE decode strategy: {window_info}")

    def _fsdp_wrap_reward_model(self):
        """
        Wrap the reward model with FSDP to shard its parameters across GPUs.
        """
        if not dist.is_initialized():
            print(f"[DMDRL] Distributed not initialized, skipping reward model FSDP")
            return

        from utils.distributed import fsdp_wrap

        print(f"[DMDRL] Wrapping reward model with FSDP...")
        self._reward_model.inferencer.model = fsdp_wrap(
            self._reward_model.inferencer.model,
            sharding_strategy=getattr(self.args, "sharding_strategy", "full"),
            mixed_precision=getattr(self.args, "mixed_precision", True),
            wrap_strategy="size",
        )
        print(f"[DMDRL] Reward model FSDP wrapping complete")

    def enable_rl(self, current_step: int) -> bool:
        if not self.rl_enabled and current_step >= self.rl_cold_start_steps:
            print(f"[DMDRL] Enabling RL at step {current_step} (cold start: {self.rl_cold_start_steps})")
            self.rl_enabled = True
            self._initialize_reward_model()
        return self.rl_enabled

    def _update_reward_stats(self, reward_value: float):
        if not self._reward_stats_initialized:
            self.rl_reward_ema_mean = reward_value
            self.rl_reward_ema_std = 1.0
            self._reward_stats_initialized = True
        else:
            delta = reward_value - self.rl_reward_ema_mean
            self.rl_reward_ema_mean = self.rl_reward_ema_mean + (1 - self.rl_reward_ema_decay) * delta
            self.rl_reward_ema_std = self.rl_reward_ema_decay * self.rl_reward_ema_std + \
                                     (1 - self.rl_reward_ema_decay) * abs(delta)
            self.rl_reward_ema_std = max(self.rl_reward_ema_std, 0.1)

    def _normalize_logits(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize raw reward logits [1, 3] using z-score from inference_config.
        Differentiable linear transform preserving gradients.
        """
        if self._reward_norm_mean is None or self._reward_norm_std is None:
            return rewards

        mean = self._reward_norm_mean.to(device=rewards.device, dtype=rewards.dtype)
        std = self._reward_norm_std.to(device=rewards.device, dtype=rewards.dtype)
        return (rewards - mean.unsqueeze(0)) / std.unsqueeze(0)

    def _sample_pixel_frames(self, video: torch.Tensor, num_frames: int = 0) -> torch.Tensor:
        """
        Uniformly sample a subset of frames from a decoded pixel video tensor.
        Ensures an even number of frames (reward model requires temporal_patch_size=2).

        Args:
            video: [T, C, H, W] video tensor (with gradient)
            num_frames: number of frames to sample (0 = use all)

        Returns:
            sampled: [T_sampled, C, H, W] sampled frames (gradient preserved)
        """
        T = video.shape[0]
        if num_frames <= 0:
            num_frames = self.rl_reward_num_frames

        if num_frames <= 0 or num_frames >= T:
            # Ensure even number of frames
            if T % 2 != 0:
                video = video[:T - 1]
            return video

        num_frames = num_frames if num_frames % 2 == 0 else num_frames - 1
        num_frames = max(num_frames, 4)

        indices = torch.linspace(0, T - 1, num_frames).round().long()
        return video[indices]

    def _decode_latent_window(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode a contiguous window of latent frames to pixel space.
        Uses gradient checkpointing to save memory.

        Args:
            latent: [B, T_window, C, H, W] latent window tensor

        Returns:
            pixel_video: [B, T_pixel, C, H, W] decoded video
        """
        def _decode(lat):
            return self.vae.decode_to_pixel(lat)

        return checkpoint_utils.checkpoint(
            _decode, latent, use_reentrant=False
        )

    def compute_rl_loss(
        self,
        latent: torch.Tensor,
        text_prompts: list,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute RL loss from generated latents.

        Pipeline:
        1. Randomly select a contiguous window of latents (synced across ranks)
        2. Decode only the window to pixel space with gradient checkpoint
        3. Optionally sub-sample pixel frames
        4. Compute reward, return negative reward as loss

        Gradients flow: reward -> pixel frames -> VAE -> latent window -> generator

        Args:
            latent: Generated latent tensor [B, T, C, H, W] (e.g. T=21)
            text_prompts: List of text prompts

        Returns:
            rl_loss: Scalar tensor representing the RL loss (negative reward)
            rl_log_dict: Dictionary containing logging information
        """
        if not self.rl_enabled or self._reward_model is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "rl_loss": 0.0,
                "rl_reward_raw": 0.0,
                "rl_reward_normalized": 0.0,
                "rl_enabled": False
            }

        batch_size = latent.shape[0]
        T_latent = latent.shape[1]  # e.g. 21

        # Step 1: Select contiguous latent window (synced across ranks)
        window_size = self.rl_latent_window_size
        if window_size > 0 and window_size < T_latent:
            max_start = T_latent - window_size
            window_start = torch.randint(0, max_start + 1, (1,), device=self.device)
            dist.broadcast(window_start, src=0)
            window_start = window_start.item()
            latent_window = latent[:, window_start:window_start + window_size, ...]
        else:
            latent_window = latent
            window_start = 0
            window_size = T_latent

        # Step 2: Decode the latent window to pixel space
        pixel_video = self._decode_latent_window(latent_window)

        # Compute rewards for each sample in the batch
        total_reward = 0.0
        raw_rewards_list = []
        normalized_rewards_list = []

        for i in range(batch_size):
            single_video = pixel_video[i]  # [T_decoded, 3, H, W]

            # Step 3: Optionally sub-sample pixel frames
            sampled_video = self._sample_pixel_frames(single_video)

            prompt = text_prompts[i] if isinstance(text_prompts, list) else text_prompts

            raw_logits = self._reward_model.compute_reward_from_vae_output(
                vae_output=sampled_video,
                prompt=prompt,
                target_height=self.rl_target_height,
                target_width=self.rl_target_width
            )

            # Apply z-score normalization from inference_config (differentiable)
            normalized_logits = self._normalize_logits(raw_logits)

            if self.rl_reward_type == "VQ":
                reward = normalized_logits[0, 0]
                raw_reward = raw_logits[0, 0].detach().item()
            elif self.rl_reward_type == "MQ":
                reward = normalized_logits[0, 1]
                raw_reward = raw_logits[0, 1].detach().item()
            elif self.rl_reward_type == "TA":
                reward = normalized_logits[0, 2]
                raw_reward = raw_logits[0, 2].detach().item()
            else:  # overall
                reward = normalized_logits.sum()
                raw_reward = raw_logits.sum().detach().item()

            reward_value = reward.detach().item()
            raw_rewards_list.append(raw_reward)
            normalized_rewards_list.append(reward_value)

            # Update EMA for monitoring
            self._update_reward_stats(reward_value)

            total_reward = total_reward + reward

        avg_reward = total_reward / batch_size
        rl_loss = -avg_reward

        rl_log_dict = {
            "rl_loss": rl_loss.detach().item(),
            "rl_reward_raw": sum(raw_rewards_list) / len(raw_rewards_list),
            "rl_reward_normalized": sum(normalized_rewards_list) / len(normalized_rewards_list),
            "rl_reward_ema_mean": self.rl_reward_ema_mean,
            "rl_reward_ema_std": self.rl_reward_ema_std,
            "rl_latent_window_start": window_start,
            "rl_latent_window_size": window_size,
            "rl_enabled": True
        }

        return rl_loss, rl_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        text_prompts: list = None,
        current_step: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        # Check if RL should be enabled
        self.enable_rl(current_step)

        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        # Step 3: Compute RL loss if enabled
        if self.rl_enabled and text_prompts is not None:
            rl_loss, rl_log_dict = self.compute_rl_loss(
                latent=pred_image,
                text_prompts=text_prompts
            )
        else:
            rl_loss = torch.tensor(0.0, device=self.device)
            rl_log_dict = {
                "rl_loss": 0.0,
                "rl_reward_mean": 0.0,
                "rl_enabled": False
            }

        # Step 4: Combine losses
        total_loss = dmd_loss + self.rl_loss_weight * rl_loss

        generator_log_dict = dmd_log_dict.copy()
        generator_log_dict.update(rl_log_dict)
        generator_log_dict["dmd_loss"] = dmd_loss.detach().item()
        generator_log_dict["total_generator_loss"] = total_loss.detach().item()
        generator_log_dict["rl_loss_weight"] = self.rl_loss_weight

        return total_loss, generator_log_dict
