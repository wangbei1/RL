"""
DMD + RL (Reinforcement Learning) Model

This module implements the combination of Distribution Matching Distillation (DMD)
with Reinforcement Learning from video reward models.

Loss: loss_gen = dmd_loss + rl_loss_weight * rl_loss

Features:
- RL loss computed from VAE-decoded videos using DifferentiableVideoReward
- Cold start: RL loss only participates after a specified number of training steps
- Reward normalization support for stable training
- Chunked VAE decoding via cached_decode: contiguous chunks with temporal cache for correct boundaries
- Pixel-space frame sampling: only a configurable subset of decoded frames is sent to the reward model
- Gradient checkpointing on reward model (Qwen2-VL)
- FSDP support for the reward model
"""

import torch
import torch.nn.functional as F
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
        self.vae_chunk_size = getattr(args, "vae_chunk_size", 0)  # 0 = no chunking

        # Frame sampling: only send a subset of frames to the reward model
        # This dramatically reduces VRAM since Qwen2-VL attention is quadratic in token count
        # 0 = no sampling (use all frames)
        self.rl_reward_num_frames = getattr(args, "rl_reward_num_frames", 10)

        # Reward normalization: use inference_config from reward model checkpoint
        # (VQ_mean/std, MQ_mean/std, TA_mean/std) for proper z-score normalization
        # These are loaded lazily when the reward model is initialized
        self._reward_norm_mean = None  # [3] tensor: [VQ_mean, MQ_mean, TA_mean]
        self._reward_norm_std = None   # [3] tensor: [VQ_std, MQ_std, TA_std]

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
        # This trades compute for memory: intermediate activations are freed during
        # forward and recomputed during backward
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
        print(f"[DMDRL] Reward model initialized and frozen successfully")
        print(f"[DMDRL] VAE chunk decode: vae_chunk_size={self.vae_chunk_size} "
              f"(0 = decode all at once)")
        print(f"[DMDRL] Pixel-space frame sampling: {self.rl_reward_num_frames} frames "
              f"(0 = all frames)")

    def _fsdp_wrap_reward_model(self):
        """
        Wrap the reward model with FSDP to shard its parameters across GPUs.
        Even though the reward model is frozen, FSDP sharding reduces per-GPU
        memory usage for the model weights.
        """
        import torch.distributed as dist
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
        Normalize raw reward logits [1, 3] using the inference_config from the
        reward model checkpoint. This is a differentiable linear transform:
            normalized[i] = (raw[i] - mean[i]) / std[i]

        After normalization, each dimension (VQ, MQ, TA) is a z-score centered
        around 0 with unit variance, making them comparable and well-scaled for RL.

        Args:
            rewards: [1, 3] raw logits tensor (VQ, MQ, TA) with gradient

        Returns:
            normalized: [1, 3] z-score normalized tensor (gradient preserved)
        """
        if self._reward_norm_mean is None or self._reward_norm_std is None:
            return rewards

        # Differentiable z-score normalization: (x - mean) / std
        mean = self._reward_norm_mean.to(device=rewards.device, dtype=rewards.dtype)
        std = self._reward_norm_std.to(device=rewards.device, dtype=rewards.dtype)
        return (rewards - mean.unsqueeze(0)) / std.unsqueeze(0)

    def _sample_frames(self, video: torch.Tensor, num_frames: int = 0) -> torch.Tensor:
        """
        Uniformly sample a subset of frames from a video tensor.
        Ensures an even number of frames (reward model requires temporal_patch_size=2).

        Args:
            video: [T, C, H, W] video tensor (with gradient)
            num_frames: number of frames to sample (0 = use self.rl_reward_num_frames)

        Returns:
            sampled: [T_sampled, C, H, W] sampled frames (gradient preserved)
        """
        T = video.shape[0]
        if num_frames <= 0:
            num_frames = self.rl_reward_num_frames

        if num_frames <= 0 or num_frames >= T:
            if T % 2 != 0:
                video = video[:T - 1]
            return video

        num_frames = num_frames if num_frames % 2 == 0 else num_frames - 1
        num_frames = max(num_frames, 4)

        indices = torch.linspace(0, T - 1, num_frames).round().long()
        return video[indices]

    def _chunk_decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode all latent frames to pixel space in contiguous temporal chunks.

        Uses the VAE's cached_decode for chunked processing: temporal conv
        caches are maintained across chunks so boundary frames are decoded
        correctly (no seam artifacts). This avoids torch.utils.checkpoint
        which fails on the VAE due to mutable internal state (_feat_map)
        causing tensor count mismatches during recomputation.

        With vae_chunk_size=N, peak VAE temporary buffer memory is proportional
        to N latent frames. Autograd activations for all chunks are retained
        for backward, but the chunking bounds peak forward allocation.

        Args:
            latent: [B, T, C, H, W] full latent tensor (e.g., T=21)

        Returns:
            pixel_video: [B, T_pixel, C, H, W] decoded video
        """
        B, T, C, H, W = latent.shape
        chunk_size = self.vae_chunk_size

        if chunk_size <= 0 or chunk_size >= T:
            # No chunking: decode all at once
            return self.vae.decode_to_pixel(latent)

        # Use cached_decode: maintains temporal conv caches across chunks
        # so chunk boundaries get correct context from the previous chunk.
        # cached_decode requires B=1; fall back to full decode otherwise.
        if B > 1:
            return self.vae.decode_to_pixel(latent)

        self.vae.model.clear_cache()

        pixel_chunks = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = latent[:, start:end]
            pixel_chunk = self.vae.decode_to_pixel(chunk, use_cache=True)
            pixel_chunks.append(pixel_chunk)

        self.vae.model.clear_cache()
        return torch.cat(pixel_chunks, dim=1)

    def compute_rl_loss(
        self,
        latent: torch.Tensor,
        text_prompts: list,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute RL loss from generated latents.

        Pipeline:
        1. Decode ALL latent frames via chunked VAE cached_decode
           (preserves temporal coherence across chunk boundaries)
        2. Sub-sample pixel frames to rl_reward_num_frames
        3. Compute reward, return negative reward as loss

        Gradients flow: reward → pixel frames → VAE → latent → generator

        Args:
            latent: Generated latent tensor [B, T, C, H, W]
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

        # Step 1: Decode ALL latent frames to pixel space in contiguous chunks
        # Each chunk is gradient-checkpointed; vae_chunk_size controls peak memory
        # e.g., vae_chunk_size=5: decode 5 latent frames at a time
        pixel_video = self._chunk_decode_latent(latent)

        # Compute rewards for each sample in the batch
        total_reward = 0.0
        raw_rewards_list = []
        normalized_rewards_list = []

        for i in range(batch_size):
            single_video = pixel_video[i]  # [T_decoded, 3, H, W]

            # Step 2: Sub-sample pixel frames for reward model
            # Full decode produces ~81 pixel frames → uniformly sample 10
            sampled_video = self._sample_frames(single_video)

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
            "rl_reward_num_frames": self.rl_reward_num_frames,
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
