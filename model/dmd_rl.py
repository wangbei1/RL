"""
DMD + RL (Reinforcement Learning) Model

This module implements the combination of Distribution Matching Distillation (DMD)
with Reinforcement Learning from video reward models.

Loss: loss_gen = dmd_loss + rl_loss_weight * rl_loss

RL approach: REINFORCE with Gaussian perturbation (zeroth-order gradient estimation).
Instead of backpropping through VAE decoder and reward model (memory-intensive),
we estimate the reward gradient via antithetic finite differences:

    nabla_x R(x) ≈ (R(x+σε) - R(x-σε)) / (2σ) * ε

Then construct a surrogate loss (same trick as DMD) whose analytical gradient
equals the estimated reward gradient. This avoids storing activations in the
VAE and reward model, significantly reducing GPU memory usage.

Features:
- REINFORCE-style RL loss with Gaussian perturbation (no backprop through reward model)
- Cold start: RL loss only participates after a specified number of training steps
- Reward EMA statistics for monitoring
- Gradient normalization for stable training
- Chunked VAE decoding (no gradient checkpointing needed) for memory efficiency
- FSDP support for the reward model
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from model.dmd import DMD


class DMDRL(DMD):
    """
    DMD + RL Model combining distribution matching distillation with
    reinforcement learning from video reward models.

    Uses REINFORCE with Gaussian perturbation for the RL loss, avoiding
    backpropagation through the VAE decoder and reward model.
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

        # REINFORCE parameters
        self.rl_perturbation_sigma = getattr(args, "rl_perturbation_sigma", 0.1)
        self.rl_num_perturbations = getattr(args, "rl_num_perturbations", 1)

        # Running statistics for reward monitoring (EMA)
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
        No gradient checkpointing needed: REINFORCE doesn't backprop through
        the reward model, so no intermediate activations need to be stored.
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

        # Freeze reward model parameters (REINFORCE: no gradients needed at all)
        self._reward_model.inferencer.model.requires_grad_(False)

        # Optional FSDP wrapping for the reward model
        if self.rl_reward_fsdp:
            self._fsdp_wrap_reward_model()

        self._reward_model_initialized = True
        print(f"[DMDRL] Reward model initialized and frozen (REINFORCE mode, no gradient checkpointing)")

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

    def _vae_decode_no_grad(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel space without gradient tracking.
        Uses temporal chunking for memory efficiency but no gradient checkpointing
        (since this runs under torch.no_grad()).

        Args:
            latent: [B, T, C, H, W] latent tensor

        Returns:
            pixel_video: [B, T, 3, H', W'] decoded pixel video
        """
        B, T, C, H, W = latent.shape
        chunk_size = self.vae_chunk_size

        if chunk_size <= 0 or chunk_size >= T:
            return self.vae.decode_to_pixel(latent)

        # Chunked decode using cached_decode to preserve temporal causality
        zs = latent.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        device, dtype = latent.device, latent.dtype
        scale = [self.vae.mean.to(device=device, dtype=dtype),
                 1.0 / self.vae.std.to(device=device, dtype=dtype)]

        decoded_chunks = []
        for b in range(B):
            single = zs[b:b+1]  # [1, C, T, H, W]
            self.vae.model.clear_cache()
            sample_chunks = []

            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk = single[:, :, start:end, :, :]
                decoded = self.vae.model.cached_decode(chunk, scale)
                decoded = decoded.float().clamp_(-1, 1)
                sample_chunks.append(decoded)

            self.vae.model.clear_cache()
            decoded_chunks.append(torch.cat(sample_chunks, dim=2))  # [1, C, T_full, H', W']

        output = torch.cat(decoded_chunks, dim=0)  # [B, C, T, H', W']
        output = output.permute(0, 2, 1, 3, 4)  # [B, T, C, H', W']
        return output

    def _compute_single_reward(self, single_video: torch.Tensor, prompt: str) -> float:
        """
        Compute scalar reward for a single video sample.
        Must be called under torch.no_grad() context.

        Args:
            single_video: [T, 3, H, W] pixel video tensor
            prompt: text prompt string

        Returns:
            Scalar reward value (float)
        """
        rewards = self._reward_model.compute_reward_from_vae_output(
            vae_output=single_video,
            prompt=prompt,
            target_height=self.rl_target_height,
            target_width=self.rl_target_width
        )

        if self.rl_reward_type == "VQ":
            return rewards[0, 0].item()
        elif self.rl_reward_type == "MQ":
            return rewards[0, 1].item()
        elif self.rl_reward_type == "TA":
            return rewards[0, 2].item()
        else:  # overall
            return rewards.sum().item()

    def _compute_batch_reward(self, latent: torch.Tensor, text_prompts: list) -> list:
        """
        Compute per-sample rewards for a batch of latents.
        Must be called under torch.no_grad() context.

        Args:
            latent: [B, T, C, H, W] latent tensor
            text_prompts: list of text prompt strings

        Returns:
            List of scalar reward values (floats), one per sample
        """
        B = latent.shape[0]
        pixel_video = self._vae_decode_no_grad(latent)

        rewards = []
        for i in range(B):
            prompt = text_prompts[i] if isinstance(text_prompts, list) else text_prompts
            r = self._compute_single_reward(pixel_video[i], prompt)
            rewards.append(r)

        return rewards

    def compute_rl_loss(
        self,
        latent: torch.Tensor,
        text_prompts: list,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute RL loss using REINFORCE with Gaussian perturbation.

        Instead of backpropping through VAE + Reward Model, we:
        1. Sample perturbation ε ~ N(0, I)
        2. Evaluate reward at (latent ± σε) under torch.no_grad() (antithetic sampling)
        3. Estimate reward gradient: ∇R ≈ (R⁺ - R⁻) / (2σ) · ε
        4. Construct surrogate loss (same trick as DMD) whose gradient = -∇R

        Minimizing the surrogate loss pushes the generator toward higher reward.

        Args:
            latent: Generated latent tensor [B, T, C, H, W] (with grad_fn from generator)
            text_prompts: List of text prompts

        Returns:
            rl_loss: Scalar surrogate loss tensor (gradient flows back to generator)
            rl_log_dict: Dictionary containing logging information
        """
        if not self.rl_enabled or self._reward_model is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "rl_loss": 0.0,
                "rl_reward_raw": 0.0,
                "rl_enabled": False
            }

        B, T, C, H, W = latent.shape
        sigma = self.rl_perturbation_sigma

        with torch.no_grad():
            latent_detached = latent.detach()
            pseudo_grad = torch.zeros_like(latent)
            all_raw_rewards = []

            for k in range(self.rl_num_perturbations):
                epsilon = torch.randn_like(latent)

                # Antithetic sampling: evaluate reward at latent ± σε
                latent_plus = latent_detached + sigma * epsilon
                latent_minus = latent_detached - sigma * epsilon

                rewards_plus = self._compute_batch_reward(latent_plus, text_prompts)
                rewards_minus = self._compute_batch_reward(latent_minus, text_prompts)

                all_raw_rewards.extend(rewards_plus)
                all_raw_rewards.extend(rewards_minus)

                # Per-sample finite difference gradient estimate:
                # ∇_x R(x) ≈ (R(x+σε) - R(x-σε)) / (2σ) · ε
                for i in range(B):
                    diff = rewards_plus[i] - rewards_minus[i]
                    pseudo_grad[i] += (diff / (2.0 * sigma)) * epsilon[i]

            pseudo_grad /= self.rl_num_perturbations

            # Update EMA reward statistics for monitoring
            mean_reward = sum(all_raw_rewards) / len(all_raw_rewards)
            self._update_reward_stats(mean_reward)

            # Normalize pseudo gradient (similar to DMD's gradient normalization)
            # This stabilizes training regardless of reward scale
            grad_norm = torch.abs(pseudo_grad).mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-8)
            pseudo_grad = pseudo_grad / grad_norm

        # Surrogate loss (same trick as DMD):
        #   loss = 0.5 * ||latent - (latent + pseudo_grad).detach()||²
        #   ∂loss/∂latent = latent - (latent + pseudo_grad)_stopped = -pseudo_grad
        #
        # Since pseudo_grad ∝ ∇R (reward increase direction),
        # optimizer step: θ -= lr * ∂loss/∂θ = lr * pseudo_grad · ∂latent/∂θ  → reward ↑
        rl_loss = 0.5 * F.mse_loss(
            latent.double(),
            (latent.double() + pseudo_grad.double()).detach(),
            reduction="mean"
        )

        rl_log_dict = {
            "rl_loss": rl_loss.detach().item(),
            "rl_reward_raw": mean_reward,
            "rl_reward_ema_mean": self.rl_reward_ema_mean,
            "rl_reward_ema_std": self.rl_reward_ema_std,
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
