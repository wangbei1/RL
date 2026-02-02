"""
DeepSpeed version of the distillation trainer.
Supports ZeRO-2/ZeRO-3 with CPU offload for memory efficiency.
"""
import gc
import logging
from datetime import datetime

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, DMDRL
import torch
import wandb
import time
import os
import json
import deepspeed


class EMA:
    """Simple EMA implementation for DeepSpeed."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class DeepSpeedTrainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # DeepSpeed handles distributed init
        deepspeed.init_distributed()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.is_main_process = dist.get_rank() == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # Random seed
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + dist.get_rank())

        # Initialize local logging
        self._init_local_logging(config)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "dmd_rl":
            self.model = DMDRL(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Load DeepSpeed config
        ds_config = self._load_deepspeed_config(config)

        # Calculate gradient accumulation steps
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        if hasattr(config, "total_batch_size"):
            effective_batch = config.batch_size * self.world_size
            self.gradient_accumulation_steps = max(1, config.total_batch_size // effective_batch)
        ds_config["gradient_accumulation_steps"] = self.gradient_accumulation_steps
        ds_config["train_micro_batch_size_per_gpu"] = config.batch_size
        ds_config["train_batch_size"] = config.batch_size * self.world_size * self.gradient_accumulation_steps

        if self.is_main_process:
            print(f"[DeepSpeed] Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"[DeepSpeed] Effective batch size: {ds_config['train_batch_size']}")

        # Step 3: Initialize DeepSpeed for generator
        generator_params = [p for p in self.model.generator.parameters() if p.requires_grad]
        self.generator_engine, self.generator_optimizer, _, _ = deepspeed.initialize(
            model=self.model.generator,
            model_parameters=generator_params,
            config=ds_config,
        )

        # Step 4: Initialize DeepSpeed for critic (fake_score)
        critic_ds_config = ds_config.copy()
        critic_ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": config.lr_critic if hasattr(config, "lr_critic") else config.lr,
                "betas": [config.beta1_critic, config.beta2_critic],
                "weight_decay": config.weight_decay
            }
        }
        critic_params = [p for p in self.model.fake_score.parameters() if p.requires_grad]
        self.critic_engine, self.critic_optimizer, _, _ = deepspeed.initialize(
            model=self.model.fake_score,
            model_parameters=critic_params,
            config=critic_ds_config,
        )

        # Move other models to device (not wrapped by DeepSpeed)
        self.model.real_score = self.model.real_score.to(self.device, dtype=self.dtype)
        self.model.real_score.eval()
        self.model.real_score.requires_grad_(False)

        self.model.text_encoder = self.model.text_encoder.to(self.device)
        self.model.text_encoder.eval()
        self.model.text_encoder.requires_grad_(False)

        # VAE
        if not config.no_visualize or config.load_raw_video or config.distribution_loss == "dmd_rl":
            self.model.vae = self.model.vae.to(self.device, dtype=self.dtype)

        # Update model references to use DeepSpeed engines
        self.model.generator = self.generator_engine.module
        self.model.fake_score = self.critic_engine.module

        # Step 5: Initialize dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: EMA setup
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0) and self.step >= config.ema_start_step:
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA(self.model.generator, decay=ema_weight)

        # Step 7: Load pretrained weights
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=True)

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def _load_deepspeed_config(self, config):
        """Load DeepSpeed config from file or use default."""
        ds_config_path = getattr(config, "deepspeed_config", "configs/deepspeed_config.json")

        if os.path.exists(ds_config_path):
            with open(ds_config_path, "r") as f:
                ds_config = json.load(f)
            if self.is_main_process:
                print(f"[DeepSpeed] Loaded config from {ds_config_path}")
        else:
            # Default ZeRO-2 config with CPU offload
            ds_config = {
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True
                },
                "bf16": {
                    "enabled": config.mixed_precision
                },
                "gradient_clipping": self.max_grad_norm_generator,
                "zero_allow_untested_optimizer": True,
                "wall_clock_breakdown": False
            }
            if self.is_main_process:
                print("[DeepSpeed] Using default ZeRO-2 config with CPU offload")

        # Add optimizer config
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "betas": [config.beta1, config.beta2],
                "weight_decay": config.weight_decay
            }
        }

        return ds_config

    def _init_local_logging(self, config):
        """Initialize local logging directory and log file."""
        if not self.is_main_process:
            self.exp_dir = None
            self.log_file = None
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = getattr(config, "config_name", "experiment")
        exp_dir_name = f"{timestamp}_{exp_name}"

        output_base = getattr(config, "output_dir", "output")
        self.exp_dir = os.path.join(output_base, exp_dir_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.log_file = os.path.join(self.exp_dir, "log.txt")

        with open(self.log_file, "w") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Distribution loss: {config.distribution_loss}\n")
            f.write(f"Backend: DeepSpeed\n")
            f.write("=" * 80 + "\n")
            if config.distribution_loss == "dmd_rl":
                f.write(f"{'step':>8} | {'dmd_loss':>10} | {'rl_loss':>10} | {'reward_raw':>12} | {'reward_norm':>12} | {'total_loss':>12} | {'rl_enabled':>10}\n")
            else:
                f.write(f"{'step':>8} | {'generator_loss':>14} | {'critic_loss':>12}\n")
            f.write("-" * 80 + "\n")

        print(f"[Logging] Experiment directory: {self.exp_dir}")
        print(f"[Logging] Log file: {self.log_file}")

    def _log_to_file(self, step, generator_log_dict, critic_log_dict):
        """Write training metrics to local log file."""
        if not self.is_main_process or self.log_file is None:
            return

        with open(self.log_file, "a") as f:
            if self.config.distribution_loss == "dmd_rl":
                dmd_loss = generator_log_dict.get("dmd_loss", 0.0)
                rl_loss = generator_log_dict.get("rl_loss", 0.0)
                reward_raw = generator_log_dict.get("rl_reward_raw", 0.0)
                reward_norm = generator_log_dict.get("rl_reward_normalized", 0.0)
                total_loss = generator_log_dict.get("total_generator_loss", 0.0)
                rl_enabled = generator_log_dict.get("rl_enabled", False)

                f.write(f"{step:>8} | {dmd_loss:>10.4f} | {rl_loss:>10.4f} | {reward_raw:>12.4f} | {reward_norm:>12.4f} | {total_loss:>12.4f} | {str(rl_enabled):>10}\n")
            else:
                gen_loss = generator_log_dict.get("generator_loss", torch.tensor(0.0))
                if isinstance(gen_loss, torch.Tensor):
                    gen_loss = gen_loss.mean().item()
                critic_loss = critic_log_dict.get("critic_loss", torch.tensor(0.0))
                if isinstance(critic_loss, torch.Tensor):
                    critic_loss = critic_loss.mean().item()

                f.write(f"{step:>8} | {gen_loss:>14.4f} | {critic_loss:>12.4f}\n")

    def save(self):
        """Save model checkpoints using DeepSpeed."""
        print("Start saving DeepSpeed checkpoint...")

        # Save generator
        generator_save_path = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}", "generator"
        )
        self.generator_engine.save_checkpoint(generator_save_path)

        # Save critic
        critic_save_path = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}", "critic"
        )
        self.critic_engine.save_checkpoint(critic_save_path)

        # Save EMA on main process
        if self.is_main_process and self.generator_ema is not None:
            ema_path = os.path.join(
                self.output_path, f"checkpoint_model_{self.step:06d}", "ema.pt"
            )
            torch.save(self.generator_ema.state_dict(), ema_path)

        if self.is_main_process:
            print(f"Checkpoint saved to {self.output_path}/checkpoint_model_{self.step:06d}/")

    def fwdbwd_one_step(self, batch, train_generator):
        """Forward and backward pass for one step."""
        self.model.generator.eval()  # prevent randomness
        self.model.fake_score.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get batch data
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Get conditional embeddings
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Generator training
        if train_generator:
            if self.config.distribution_loss == "dmd_rl":
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent if self.config.i2v else None,
                    text_prompts=text_prompts,
                    current_step=self.step
                )
            else:
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=image_latent if self.config.i2v else None
                )

            # DeepSpeed handles backward and gradient clipping
            self.generator_engine.backward(generator_loss)

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": torch.tensor(0.0)  # DeepSpeed handles clipping internally
            })

            return generator_log_dict

        # Step 4: Critic training
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        self.critic_engine.backward(critic_loss)

        critic_log_dict.update({
            "critic_loss": critic_loss,
            "critic_grad_norm": torch.tensor(0.0)
        })

        return critic_log_dict

    def train(self):
        """Main training loop."""
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                extras_list = []
                for _ in range(self.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)

                generator_log_dict = merge_dict_list(extras_list)
                self.generator_engine.step()

                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            extras_list = []
            for _ in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)

            critic_log_dict = merge_dict_list(extras_list)
            self.critic_engine.step()

            # Increment step
            self.step += 1

            # Create EMA if needed
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA(self.model.generator, decay=self.config.ema_weight)

            # Save checkpoint
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update({
                        "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                        "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                        "dmdtrain_gradient_norm": generator_log_dict.get("dmdtrain_gradient_norm", torch.tensor(0.0)).mean().item()
                    })

                    if self.config.distribution_loss == "dmd_rl":
                        rl_metrics = [
                            "rl_loss", "rl_reward_raw", "rl_reward_normalized",
                            "rl_reward_ema_mean", "rl_reward_ema_std", "rl_enabled",
                            "dmd_loss", "total_generator_loss"
                        ]
                        for key in rl_metrics:
                            if key in generator_log_dict:
                                value = generator_log_dict[key]
                                wandb_loss_dict[key] = float(value) if isinstance(value, bool) else value

                wandb_loss_dict.update({
                    "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                    "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                })

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                if TRAIN_GENERATOR:
                    self._log_to_file(self.step, generator_log_dict, critic_log_dict)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
