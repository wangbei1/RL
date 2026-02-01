"""
DMD + RL (Reinforcement Learning) Model

This module implements the combination of Distribution Matching Distillation (DMD)
with Reinforcement Learning from video reward models.

Loss: loss_gen = dmd_loss + rl_loss_weight * rl_loss

Features:
- RL loss computed from VAE-decoded videos using DifferentiableVideoReward
- Cold start: RL loss only participates after a specified number of training steps
- LoRA support for the generator to prevent training collapse
"""

from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.dmd import DMD


class DMDRL(DMD):
    """
    DMD + RL Model combining distribution matching distillation with
    reinforcement learning from video reward models.
    """

    def __init__(self, args, device):
        """
        Initialize the DMD+RL module.

        Additional args (compared to DMD):
            - rl_loss_weight: Weight for RL loss (default: 1.0)
            - rl_cold_start_steps: Number of steps before RL loss is enabled (default: 0)
            - rl_reward_checkpoint: Path to the reward model checkpoint
            - rl_target_height: Target height for reward model (must be multiple of 28)
            - rl_target_width: Target width for reward model (must be multiple of 28)
            - rl_reward_type: Type of reward to use ('VQ', 'MQ', 'TA', 'overall') (default: 'overall')
        """
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

        # Validate target dimensions
        assert self.rl_target_height % 28 == 0, f"rl_target_height must be multiple of 28, got {self.rl_target_height}"
        assert self.rl_target_width % 28 == 0, f"rl_target_width must be multiple of 28, got {self.rl_target_width}"

        # Lazy initialization of reward model (to save memory during cold start)
        self._reward_model = None
        self._reward_model_initialized = False

    def _initialize_reward_model(self):
        """
        Lazily initialize the reward model when RL is first enabled.
        This saves GPU memory during the cold start period.
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
        self._reward_model_initialized = True
        print(f"[DMDRL] Reward model initialized successfully")

    def enable_rl(self, current_step: int) -> bool:
        """
        Check and enable RL training if cold start period has passed.

        Args:
            current_step: Current training step

        Returns:
            True if RL is enabled, False otherwise
        """
        if not self.rl_enabled and current_step >= self.rl_cold_start_steps:
            print(f"[DMDRL] Enabling RL at step {current_step} (cold start: {self.rl_cold_start_steps})")
            self.rl_enabled = True
            self._initialize_reward_model()
        return self.rl_enabled

    def compute_rl_loss(
        self,
        latent: torch.Tensor,
        text_prompts: list,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute RL loss from generated latents.

        Args:
            latent: Generated latent tensor [B, T, C, H, W]
            text_prompts: List of text prompts

        Returns:
            rl_loss: Scalar tensor representing the RL loss (negative reward)
            rl_log_dict: Dictionary containing logging information
        """
        if not self.rl_enabled or self._reward_model is None:
            # Return zero loss if RL is not enabled
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "rl_loss": 0.0,
                "rl_reward_mean": 0.0,
                "rl_enabled": False
            }

        batch_size = latent.shape[0]

        # Decode latent to pixel space
        # latent: [B, T, C, H, W] -> pixel: [B, T, 3, H*8, W*8]
        pixel_video = self.vae.decode_to_pixel(latent)

        # Compute rewards for each sample in the batch
        total_reward = 0.0
        rewards_list = []

        for i in range(batch_size):
            # Get single video: [T, C, H, W]
            single_video = pixel_video[i]  # [T, 3, H, W]
            prompt = text_prompts[i] if isinstance(text_prompts, list) else text_prompts

            # Compute reward using the differentiable path
            # rewards shape: [1, 3] (VQ, MQ, TA)
            rewards = self._reward_model.compute_reward_from_vae_output(
                vae_output=single_video,
                prompt=prompt,
                target_height=self.rl_target_height,
                target_width=self.rl_target_width
            )

            # Select which reward to use
            if self.rl_reward_type == "VQ":
                reward = rewards[0, 0]
            elif self.rl_reward_type == "MQ":
                reward = rewards[0, 1]
            elif self.rl_reward_type == "TA":
                reward = rewards[0, 2]
            else:  # overall
                reward = rewards.sum()

            total_reward = total_reward + reward
            rewards_list.append(reward.detach().item())

        # Average reward across batch
        avg_reward = total_reward / batch_size

        # RL loss is negative reward (we want to maximize reward)
        rl_loss = -avg_reward

        rl_log_dict = {
            "rl_loss": rl_loss.detach().item(),
            "rl_reward_mean": sum(rewards_list) / len(rewards_list),
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
        """
        Generate image/videos from noise and compute the combined DMD + RL loss.

        Additional args (compared to DMD):
            - text_prompts: List of text prompts for RL reward computation
            - current_step: Current training step for cold start check

        Output:
            - loss: Combined loss (dmd_loss + rl_loss_weight * rl_loss)
            - generator_log_dict: Dictionary containing intermediate tensors for logging
        """
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
        # loss_gen = dmd_loss + rl_loss_weight * rl_loss
        total_loss = dmd_loss + self.rl_loss_weight * rl_loss

        # Merge log dicts
        generator_log_dict = dmd_log_dict.copy()
        generator_log_dict.update(rl_log_dict)
        generator_log_dict["dmd_loss"] = dmd_loss.detach().item()
        generator_log_dict["total_generator_loss"] = total_loss.detach().item()
        generator_log_dict["rl_loss_weight"] = self.rl_loss_weight

        return total_loss, generator_log_dict
