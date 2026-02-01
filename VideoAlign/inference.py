import ast
import json
import os
import pdb
from collections.abc import Mapping
import pandas as pd
from torchvision.transforms import InterpolationMode

import torch
from vision_process import process_vision_info, smart_resize
from torchvision import io, transforms

from data import DataConfig
from utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from utils import load_model_from_checkpoint
from train_reward import create_model_and_processor
from prompt_template import build_prompt


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return config_dict["data_config"], None, config_dict["model_config"], config_dict["peft_lora_config"], \
           config_dict["inference_config"] if "inference_config" in config_dict else None

class VideoVLMRewardInference():
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def debug_print_shapes(self, data, name="Data", indent=0):
        """
        递归打印数据结构，包括张量的形状、均值、方差、极值和部分取值。
        """
        tab = "  " * indent
        if indent == 0:
            print(f"\n{'#'*30} DEBUG: {name} {'#'*30}")

        if isinstance(data, (dict, Mapping)):
            print(f"{tab}Dict (keys: {list(data.keys())})")
            for k, v in data.items():
                print(f"{tab}Key: '{k}'")
                self.debug_print_shapes(v, name=k, indent=indent + 1)

        elif isinstance(data, (list, tuple)):
            print(f"{tab}List/Tuple (length: {len(data)})")
            if len(data) > 0:
                # 如果是列表且元素很多，只展示第一个和最后一个样本的详细信息，避免刷屏
                if len(data) > 2:
                    print(f"{tab}  [Showing first item]:")
                    self.debug_print_shapes(data[0], name="list[0]", indent=indent + 1)
                    print(f"{tab}  ...")
                    print(f"{tab}  [Showing last item]:")
                    self.debug_print_shapes(data[-1], name=f"list[{len(data)-1}]", indent=indent + 1)
                else:
                    for i, item in enumerate(data):
                        self.debug_print_shapes(item, name=f"list[{i}]", indent=indent + 1)

        elif isinstance(data, torch.Tensor):
            # 获取基本属性
            shape = list(data.shape)
            dtype = data.dtype
            device = data.device
            
            # 计算统计量 (仅对数值型张量)
            with torch.no_grad():
                if data.numel() > 0 and torch.is_floating_point(data):
                    v_min = data.min().item()
                    v_max = data.max().item()
                    v_mean = data.mean().item()
                    v_std = data.std().item()
                elif data.numel() > 0: # 整数类型 (如 input_ids)
                    v_min = data.min().item()
                    v_max = data.max().item()
                    v_mean = data.float().mean().item()
                    v_std = data.float().std().item()
                else:
                    v_min = v_max = v_mean = v_std = 0

            # 获取前几个数据点作为例子
            flat_data = data.flatten()
            sample_size = min(5, flat_data.numel())
            samples = flat_data[:sample_size].tolist()
            sample_str = ", ".join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in samples])

            print(f"{tab}Tensor Shape: {shape}")
            print(f"{tab}  - Dtype: {dtype} | Device: {device}")
            print(f"{tab}  - Stats: Min: {v_min:.4f}, Max: {v_max:.4f}, Mean: {v_mean:.4f}, Std: {v_std:.4f}")
            print(f"{tab}  - Samples: [{sample_str}...]")

        else:
            print(f"{tab}Value: {data} (Type: {type(data)})")

        if indent == 0:
            print(f"{'#'*80}\n")
    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)
        print(">>> 检查 process_vision_info 输出:")
        self.debug_print_shapes(image_inputs, "image_inputs")
        self.debug_print_shapes(video_inputs, "video_inputs")

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        print(">>> 检查 self.processor 后的 batch (CPU):")
        self.debug_print_shapes(batch, "batch_pre_move")
        batch = self._prepare_inputs(batch)
                # --- 调试点 3: _prepare_inputs 后的 batch (此时在 GPU 上) ---
        print(">>> 检查 self._prepare_inputs 后的 batch (GPU):")
        self.debug_print_shapes(batch, "batch_post_move")
        return batch

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]
        print("Raw rewards:", rewards)

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards

import torch
import torch.nn.functional as F

import torch

import torch
import itertools

import torch
import torch.nn.functional as F
import itertools
from vision_process import process_vision_info

import torch

import torch
import torch.nn.functional as F
import itertools
from vision_process import process_vision_info

import torch
import torch.nn.functional as F
import itertools
from vision_process import process_vision_info

import torch
import torch.nn.functional as F

def differentiable_process_vlm_video_v446(video_tensor, processor):
    """
    严格按照 Qwen2-VL v4.46.2 官方 9 维拆解逻辑实现
    video_tensor: [T, C, H, W], float32, [0, 255]
    """
    # 1. 归一化参数获取
    image_processor = processor.image_processor
    mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1).to(video_tensor.device)
    std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1).to(video_tensor.device)

    # 2. 归一化
    x = video_tensor / 255.0
    x = (x - mean) / std

    # 3. 基础维度定义
    T, C, H, W = x.shape
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2  # 官方核心参数
    
    grid_t = T // temporal_patch_size
    grid_h = H // patch_size
    grid_w = W // patch_size

    # 4. 9 维拆解 (严格对齐官方 reshape 逻辑)
    # 官方顺序：grid_t, ts, channel, gh//m, m, ps, gw//m, m, ps
    # 注意：PyTorch 默认是 C-first，所以 view 顺序需极其精确
    x = x.view(
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    # 官方代码：patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    # 对应物理含义：
    # 0: grid_t
    # 3: grid_h // 2
    # 6: grid_w // 2
    # 4: merge_h (2)
    # 7: merge_w (2)
    # 2: channel (3)
    # 1: ts (2)
    # 5: patch_h (14)
    # 8: patch_w (14)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

    # 外部：grid_t * (grid_h/2) * (grid_w/2) * 2 * 2 = grid_t * grid_h * grid_w
    # 内部：3 * 2 * 14 * 14 = 1176
    pixel_values_videos = x.view(-1, C * temporal_patch_size * patch_size * patch_size)
    
    return pixel_values_videos

import torch
import torch.nn.functional as F
from vision_process import process_vision_info

def verify_v446_consistency(inferencer, video_path, prompt):
    print(">>> 正在获取官方输出...")
    # 1. 官方 Processor 处理
    batch_official = inferencer.prepare_batch([video_path], [prompt], num_frames=10)
    gt_pixels = batch_official['pixel_values_videos'].cpu().float()
    grid_thw = batch_official['video_grid_thw'][0].tolist()
    
    # 2. 获取原始输入并 Resize 到官方目标尺寸
    target_h, target_w = grid_thw[1] * 14, grid_thw[2] * 14
    chat_data = [[{"role": "user", "content": [{"type": "video", "video": f"file://{video_path}", "nframes": 10, "sample_type": inferencer.data_config.sample_type}, {"type": "text", "text": prompt}]}]]
    _, video_inputs = process_vision_info(chat_data)
    video_tensor_raw = video_inputs[0].clone().detach().float().to(inferencer.device)
    
    # 确保尺寸完全一致（由官方 Grid 反推）
    video_tensor_raw = F.interpolate(video_tensor_raw, size=(target_h, target_w), mode='bicubic', align_corners=False)
    video_tensor_raw.requires_grad = True
    print(video_tensor_raw.shape)

    # 3. 运行新版 9 维对齐函数
    print(">>> 运行 9 维对齐可微处理...")
    my_pixels = differentiable_process_vlm_video_v446(video_tensor_raw, inferencer.processor)
    my_pixels_cpu = my_pixels.detach().cpu()

    # 4. 对比结果
    mse = torch.mean((gt_pixels - my_pixels_cpu)**2).item()
    max_diff = torch.max(torch.abs(gt_pixels - my_pixels_cpu)).item()
    
    print("\n" + "="*50)
    print(f"MSE: {mse:.10f}")
    print(f"Max Diff: {max_diff:.10f}")
    
    if mse < 1e-5:
        print("✅ 完美对齐！终于找到了官方 v4.46.2 的真实逻辑。")
        # 梯度测试
        my_pixels.sum().backward()
        print(f"✅ 梯度检查通过: Grad Mean = {video_tensor_raw.grad.abs().mean().item():.6f}")
    else:
        print("❌ 依然不对，请检查官方 Processor 的具体版本是否带有其他自定义转换。")
        print(f"官方前5: {gt_pixels[0, :5].tolist()}")
        print(f"手动前5: {my_pixels_cpu[0, :5].tolist()}")
    print("="*50)
from torchvision import io


def get_mock_vae_output(video_path, num_frames=100, height=480, width=832, device="cuda"):
    """
    读取视频并转换为模拟 VAE Decoder 输出的张量 [-1, 1]
    """
    # 1. 读取视频帧 [T, H, W, C]
    # pts_unit='sec' 确保时间戳对齐
    vframes, _, _ = io.read_video(video_path, pts_unit='sec', output_format="TCHW")

    # 2. 均匀采样帧数
    total_frames = vframes.shape[0]
    indices = torch.linspace(0, total_frames - 1, steps=num_frames).long()
    sampled_frames = vframes[indices] # [num_frames, C, H, W]

    # 3. 转换为 float32 并 Resize 到 VAE 通常输出的分辨率
    # 注意：VAE 输出的分辨率通常是 8 的倍数
    video_tensor = sampled_frames.to(device).float()

    video_tensor = F.interpolate(video_tensor, size=(height, width), mode='bicubic', align_corners=False)

    # 4. 核心步骤：归一化到 [-1, 1]
    # 公式：(x / 127.5) - 1.0
    video_vae_style = (video_tensor / 127.5) - 1.0

    # 5. 开启梯度，模拟训练状态
    video_vae_style.requires_grad = True

    return video_vae_style


def get_video_tensor_for_reward(video_path, num_frames, max_pixels, sample_type="uniform", device="cuda"):
    """
    严格按照官方 fetch_video 逻辑读取视频，返回可微分的张量。

    这个函数的输出与 process_vision_info 完全一致，但保持 requires_grad=True。

    Args:
        video_path: 视频路径
        num_frames: 采样帧数
        max_pixels: 每帧最大像素数
        sample_type: 采样类型 ("uniform" 或 "multi_pts")
        device: 设备

    Returns:
        video_tensor: [T, C, H, W] float32 张量，范围 [0, 255]，可求梯度
        resized_height: resize 后的高度
        resized_width: resize 后的宽度
    """
    from vision_process import smart_resize, round_by_factor, FRAME_FACTOR

    # 1. 读取视频
    vframes, _, info = io.read_video(video_path, pts_unit='sec', output_format="TCHW")
    total_frames = vframes.shape[0]
    video_fps = info.get("video_fps", 30.0)

    # 2. 帧数处理 - 严格对齐官方逻辑
    nframes = round_by_factor(num_frames, FRAME_FACTOR)
    if nframes > total_frames:
        nframes = total_frames

    # 3. 帧采样 - 使用 .round().long() 与官方一致
    if sample_type == 'uniform':
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    elif sample_type == 'multi_pts':
        frames_each_pts = 6
        num_pts = 4
        fps = 8
        nframes_temp = int(total_frames * fps // video_fps)
        frames_idx = torch.linspace(0, total_frames - 1, nframes_temp).round().long().tolist()
        start_pt = int(frames_each_pts // 2)
        end_pt = int(nframes_temp - frames_each_pts // 2 - 1)
        pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
        idx = []
        for pt in pts:
            idx.extend(frames_idx[pt - frames_each_pts // 2 : pt + frames_each_pts // 2])
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    video = vframes[idx]  # [T, C, H, W]
    nframes, _, height, width = video.shape

    # 4. 计算 resize 尺寸 - 严格按照 fetch_video 逻辑
    VIDEO_MIN_PIXELS = 128 * 28 * 28
    VIDEO_MAX_PIXELS = 768 * 28 * 28
    VIDEO_TOTAL_PIXELS = 24576 * 28 * 28

    min_pixels = VIDEO_MIN_PIXELS
    total_pixels = VIDEO_TOTAL_PIXELS
    computed_max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    final_max_pixels = max_pixels if max_pixels is not None else computed_max_pixels

    resized_height, resized_width = smart_resize(
        height, width,
        factor=28,
        min_pixels=min_pixels,
        max_pixels=final_max_pixels,
    )

    # 5. Resize - 与官方完全一致
    video_tensor = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float().to(device)

    # 6. 开启梯度
    video_tensor.requires_grad = True

    return video_tensor, resized_height, resized_width


def vae_output_to_pixel_values(vae_output, target_height, target_width, processor):
    """
    将 VAE 输出 ([-1, 1]) 转换为 VLM 所需的 pixel_values。

    完整的可微分流水线：VAE output -> resize -> normalize -> 9维拆解

    Args:
        vae_output: [T, C, H, W] 张量，范围 [-1, 1]，来自 VAE decoder
        target_height: 目标高度 (必须是 28 的倍数)
        target_width: 目标宽度 (必须是 28 的倍数)
        processor: Qwen2-VL processor

    Returns:
        pixel_values_videos: [N, 1176] 格式的 token 张量
    """
    # 1. 转回 [0, 255] 范围
    video_255 = (vae_output + 1.0) * 127.5

    # 2. Resize 到目标尺寸 (只做一次 resize)
    video_resized = transforms.functional.resize(
        video_255,
        [target_height, target_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    # 3. 9 维对齐处理
    pixel_values = differentiable_process_vlm_video_v446(video_resized, processor)

    return pixel_values


import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def compare_manual_vs_official_rewards(inferencer, video_path, prompt):
    print("\n" + "="*25 + " 开始双路径对比验证 " + "="*25)
    device = inferencer.device
    model_dtype = inferencer.model.dtype

    # ---------------------------------------------------------
    # 第一部分：官方路径 (Official Path)
    # ---------------------------------------------------------
    with torch.no_grad():
        # 1. 使用官方 processor 得到 batch
        batch_official = inferencer.prepare_batch([video_path], [prompt], num_frames=10)
        # 2. 官方推理得到奖励
        rewards_official = inferencer.model(return_dict=True, **batch_official)["logits"]
    
    gt_pixels = batch_official['pixel_values_videos'].float().cpu()
    grid_thw = batch_official['video_grid_thw']

    # ---------------------------------------------------------
    # 第二部分：手动可微路径 (Manual Differentiable Path)
    # ---------------------------------------------------------
    # 1. 模拟 VAE 输出并转回 [0, 255]
    # 注意：为了公平对比，我们直接从 get_mock_vae_output 拿到张量
    vae_output = get_mock_vae_output(video_path, num_frames=10)
    vae_output_255 = (vae_output + 1) / 2 * 255.0
    
    # 2. 严格按照官方 Grid 进行 Resize
    target_h, target_w = grid_thw[0, 1].item() * 14, grid_thw[0, 2].item() * 14
    video_resized = transforms.functional.resize(
        vae_output_255,
        [target_h, target_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    )

    # 3. 运行 9 维对齐的可微处理函数
    pixel_values_manual = differentiable_process_vlm_video_v446(video_resized, inferencer.processor)
    
    # 4. 构造手动 Batch 并推理
    # 我们保留梯度，模拟真实训练场景
    input_batch_manual = {
        "input_ids": batch_official["input_ids"],
        "attention_mask": batch_official["attention_mask"],
        "video_grid_thw": batch_official["video_grid_thw"],
        "pixel_values_videos": pixel_values_manual.to(model_dtype)
    }
    
    # 手动路径推理
    outputs_manual = inferencer.model(return_dict=True, **input_batch_manual)
    rewards_manual = outputs_manual["logits"]

    # ---------------------------------------------------------
    # 第三部分：差异分析
    # ---------------------------------------------------------
    # 1. Pixel 差异
    pixel_manual_cpu = pixel_values_manual.detach().float().cpu()
    pixel_mse = torch.mean((gt_pixels - pixel_manual_cpu)**2).item()
    pixel_max_diff = torch.max(torch.abs(gt_pixels - pixel_manual_cpu)).item()

    # 2. Reward Logits 差异 (VQ, MQ, TA)
    # 将结果转回 float32 方便对比
    res_off = rewards_official.float().cpu()[0]
    res_man = rewards_manual.detach().float().cpu()[0]
    logit_diff = torch.abs(res_off - res_man)

    print("\n[维度验证]")
    print(f"Pixel Values Shape: {gt_pixels.shape}")
    print(f"Grid THW: {grid_thw.tolist()}")

    print("\n[1. 输入特征 Pixel Values 差异]")
    print(f"MSE: {pixel_mse:.10f}")
    print(f"Max Absolute Diff: {pixel_max_diff:.10f}")

    print("\n[2. 最终 Reward Logits (未归一化) 对比]")
    dims = ['VQ', 'MQ', 'TA']
    for i, name in enumerate(dims):
        print(f"{name} Dimension: Official={res_off[i]:.4f} | Manual={res_man[i]:.4f} | Diff={logit_diff[i]:.4f}")

    reward_mse = torch.mean(logit_diff**2).item()
    print(f"\nReward Logits Total MSE: {reward_mse:.10e}")

    if reward_mse < 1e-3:
        print("\n✅ 验证通过：手动路径推理结果与官方原版几乎无异，可以放心用于训练。")
    else:
        print("\n⚠️ 警告：推理结果存在偏差，请检查 Resize 逻辑或归一化参数。")

    return rewards_manual

import random
from tqdm import tqdm


def batch_validate_reward_consistency_v2(inferencer, video_dir, num_samples=100):
    """
    修复后的批量验证函数 - 确保官方路径和手动路径完全一致。

    核心修复：
    1. 使用 get_video_tensor_for_reward 与官方帧采样和 resize 逻辑完全一致
    2. 避免两次 resize 导致的累积误差
    3. 使用 .round().long() 与官方帧索引计算一致
    """
    print(f"\n开始批量验证 (V2 - 修复版)... 目标样本数: {num_samples}")

    all_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if len(all_videos) < num_samples:
        print(f"警告: 视频数量不足 {num_samples}，将处理所有 {len(all_videos)} 个视频。")
        sample_videos = all_videos
    else:
        sample_videos = random.sample(all_videos, num_samples)

    results = []
    num_frames = inferencer.data_config.num_frames if inferencer.data_config.num_frames else 10
    max_pixels = inferencer.data_config.max_frame_pixels

    for vid_name in tqdm(sample_videos):
        video_path = os.path.join(video_dir, vid_name)
        prompt = "The video shows natural movement and high quality scene."

        try:
            with torch.no_grad():
                # --- 官方路径 ---
                batch_off = inferencer.prepare_batch([video_path], [prompt], num_frames=num_frames)
                res_off = inferencer.model(**batch_off)["logits"].float().cpu()[0]
                gt_pixels = batch_off['pixel_values_videos'].float().cpu()

            # --- 手动可微路径 (使用修复后的函数) ---
            # 1. 使用与官方完全一致的视频读取和 resize 逻辑
            video_tensor, resized_h, resized_w = get_video_tensor_for_reward(
                video_path,
                num_frames=num_frames,
                max_pixels=max_pixels,
                sample_type=inferencer.data_config.sample_type,
                device=inferencer.device
            )

            # 2. 直接进行 9 维处理 (不需要额外 resize，因为 get_video_tensor_for_reward 已经处理好)
            pixel_values_manual = differentiable_process_vlm_video_v446(video_tensor, inferencer.processor)

            # 3. 构造 batch 并推理
            batch_man = {
                "input_ids": batch_off["input_ids"],
                "attention_mask": batch_off["attention_mask"],
                "video_grid_thw": batch_off["video_grid_thw"],
                "pixel_values_videos": pixel_values_manual.to(inferencer.model.dtype)
            }

            with torch.no_grad():
                res_man = inferencer.model(**batch_man)["logits"].float().cpu()[0]

            # 4. 检查 pixel 差异
            pixel_manual_cpu = pixel_values_manual.detach().float().cpu()
            pixel_mse = torch.mean((gt_pixels - pixel_manual_cpu)**2).item()

            # 5. 记录数据
            diff = torch.abs(res_off - res_man)
            results.append({
                'video': vid_name,
                'pixel_mse': pixel_mse,
                'off_VQ': res_off[0].item(), 'man_VQ': res_man[0].item(),
                'off_MQ': res_off[1].item(), 'man_MQ': res_man[1].item(),
                'off_TA': res_off[2].item(), 'man_TA': res_man[2].item(),
                'diff_VQ': diff[0].item(), 'diff_MQ': diff[1].item(), 'diff_TA': diff[2].item(),
                'sign_match_VQ': (res_off[0] * res_man[0] > 0).item(),
                'sign_match_MQ': (res_off[1] * res_man[1] > 0).item(),
                'sign_match_TA': (res_off[2] * res_man[2] > 0).item()
            })

        except Exception as e:
            print(f"跳过视频 {vid_name}, 错误: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(results)

    print("\n" + "="*20 + " 批量验证统计报告 (V2) " + "="*20)
    stats = {
        "平均 Pixel MSE": df['pixel_mse'].mean(),
        "平均绝对误差 (MAE) - VQ": df['diff_VQ'].mean(),
        "平均绝对误差 (MAE) - MQ": df['diff_MQ'].mean(),
        "平均绝对误差 (MAE) - TA": df['diff_TA'].mean(),
        "正负号一致率 (VQ)": df['sign_match_VQ'].mean(),
        "正负号一致率 (MQ)": df['sign_match_MQ'].mean(),
        "正负号一致率 (TA)": df['sign_match_TA'].mean(),
        "最大绝对偏差 (MaxDiff)": df[['diff_VQ', 'diff_MQ', 'diff_TA']].max().max()
    }

    for k, v in stats.items():
        print(f"{k:30}: {v:.6f}")

    print("\n[相关性分析]")
    print(f"VQ 相关系数: {df['off_VQ'].corr(df['man_VQ']):.4f}")
    print(f"MQ 相关系数: {df['off_MQ'].corr(df['man_MQ']):.4f}")
    print(f"TA 相关系数: {df['off_TA'].corr(df['man_TA']):.4f}")

    return df


def batch_validate_reward_consistency(inferencer, video_dir, num_samples=100):
    """
    原始验证函数 - 保留用于对比，但推荐使用 batch_validate_reward_consistency_v2
    """
    print(f"\n开始批量验证... 目标样本数: {num_samples}")

    # 1. 获取目录下所有视频文件
    all_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if len(all_videos) < num_samples:
        print(f"警告: 视频数量不足 {num_samples}，将处理所有 {len(all_videos)} 个视频。")
        sample_videos = all_videos
    else:
        sample_videos = random.sample(all_videos, num_samples)

    # 2. 初始化统计列表
    results = []

    # 3. 循环测试
    for vid_name in tqdm(sample_videos):
        video_path = os.path.join(video_dir, vid_name)
        # 这里可以使用动态 Prompt，或者固定一个通用 Prompt
        prompt = "The video shows natural movement and high quality scene."

        try:
            # 使用之前的对比逻辑获取结果 (注意: compare_manual_vs_official_rewards 需要稍微改动返回 diff 数据)
            # 为了加速，我们在 compare 内部用 torch.no_grad()
            with torch.no_grad():
                # --- 官方路径 ---
                batch_off = inferencer.prepare_batch([video_path], [prompt], num_frames=10)
                res_off = inferencer.model(**batch_off)["logits"].float().cpu()[0]

                # --- 手动路径 ---
                # 模拟 VAE 输出
                vae_out = get_mock_vae_output(video_path, num_frames=10)
                vae_255 = (vae_out + 1) / 2 * 255.0

                # 匹配尺寸并 Resize
                grid_thw = batch_off['video_grid_thw'][0]
                th, tw = grid_thw[1] * 14, grid_thw[2] * 14
                v_res = transforms.functional.resize(vae_255, [th, tw],
                                                     interpolation=InterpolationMode.BICUBIC, antialias=True)

                # 9维处理
                pix_man = differentiable_process_vlm_video_v446(v_res, inferencer.processor)

                # 推理
                batch_man = {
                    "input_ids": batch_off["input_ids"],
                    "attention_mask": batch_off["attention_mask"],
                    "video_grid_thw": batch_off["video_grid_thw"],
                    "pixel_values_videos": pix_man.to(inferencer.model.dtype)
                }
                res_man = inferencer.model(**batch_man)["logits"].float().cpu()[0]

            # 4. 记录数据
            diff = torch.abs(res_off - res_man)
            results.append({
                'video': vid_name,
                'off_VQ': res_off[0].item(), 'man_VQ': res_man[0].item(),
                'off_MQ': res_off[1].item(), 'man_MQ': res_man[1].item(),
                'off_TA': res_off[2].item(), 'man_TA': res_man[2].item(),
                'diff_VQ': diff[0].item(), 'diff_MQ': diff[1].item(), 'diff_TA': diff[2].item(),
                'sign_match_VQ': (res_off[0] * res_man[0] > 0).item(),
                'sign_match_MQ': (res_off[1] * res_man[1] > 0).item(),
                'sign_match_TA': (res_off[2] * res_man[2] > 0).item()
            })

        except Exception as e:
            print(f"跳过视频 {vid_name}, 错误: {e}")

    # 5. 聚合统计分析
    df = pd.DataFrame(results)

    print("\n" + "="*20 + " 批量验证统计报告 " + "="*20)
    stats = {
        "平均绝对误差 (MAE) - VQ": df['diff_VQ'].mean(),
        "平均绝对误差 (MAE) - MQ": df['diff_MQ'].mean(),
        "平均绝对误差 (MAE) - TA": df['diff_TA'].mean(),
        "正负号一致率 (VQ)": df['sign_match_VQ'].mean(),
        "正负号一致率 (MQ)": df['sign_match_MQ'].mean(),
        "正负号一致率 (TA)": df['sign_match_TA'].mean(),
        "最大绝对偏差 (MaxDiff)": df[['diff_VQ', 'diff_MQ', 'diff_TA']].max().max()
    }

    for k, v in stats.items():
        print(f"{k:25}: {v:.4f}")

    # 相关性分析 (Pearson Correlation)
    print("\n[相关性分析]")
    print(f"VQ 相关系数: {df['off_VQ'].corr(df['man_VQ']):.4f}")
    print(f"MQ 相关系数: {df['off_MQ'].corr(df['man_MQ']):.4f}")
    print(f"TA 相关系数: {df['off_TA'].corr(df['man_TA']):.4f}")

    return df

class DifferentiableVideoReward:
    """
    可微分的视频奖励推理类，用于 RL 训练。

    使用方法：
    ```python
    reward_model = DifferentiableVideoReward("./checkpoints", device="cuda:0")

    # 从 VAE decoder 输出计算奖励 (保持梯度)
    vae_output = vae.decode(latent)  # [-1, 1] 范围
    reward = reward_model.compute_reward_from_vae_output(
        vae_output,
        prompt="A cat walking on grass",
        target_height=336,  # 必须是 28 的倍数
        target_width=504,   # 必须是 28 的倍数
    )
    # reward 形状: [1, 3] (VQ, MQ, TA)
    # 可以直接 .backward()
    loss = -reward.sum()
    loss.backward()
    ```
    """

    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            device=device,
            dtype=dtype
        )
        self.device = device
        self.dtype = dtype

    def compute_reward_from_vae_output(self, vae_output, prompt, target_height, target_width):
        """
        从 VAE 输出计算可微分的奖励。

        Args:
            vae_output: [T, C, H, W] 张量，范围 [-1, 1]，来自 VAE decoder
            prompt: 文本提示
            target_height: 目标高度 (必须是 28 的倍数)
            target_width: 目标宽度 (必须是 28 的倍数)

        Returns:
            rewards: [1, 3] 张量 (VQ, MQ, TA)，保持梯度
        """
        assert target_height % 28 == 0, f"target_height 必须是 28 的倍数，当前为 {target_height}"
        assert target_width % 28 == 0, f"target_width 必须是 28 的倍数，当前为 {target_width}"

        # 1. 转换 VAE 输出到 pixel_values
        pixel_values = vae_output_to_pixel_values(
            vae_output,
            target_height,
            target_width,
            self.inferencer.processor
        )

        # 2. 计算 video_grid_thw
        T = vae_output.shape[0]
        grid_t = T // 2  # temporal_patch_size = 2
        grid_h = target_height // 14  # patch_size = 14
        grid_w = target_width // 14
        video_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=self.device)

        # 3. 获取文本 tokens（需要传入 grid 信息以正确计算视频 token 数量）
        text_batch = self._prepare_text_tokens(prompt, grid_t, grid_h, grid_w)

        # 4. 构造完整 batch
        batch = {
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "video_grid_thw": video_grid_thw,
            "pixel_values_videos": pixel_values.to(self.dtype)
        }

        # 5. 前向传播
        outputs = self.inferencer.model(return_dict=True, **batch)
        rewards = outputs["logits"]

        return rewards

    def _prepare_text_tokens(self, prompt, grid_t, grid_h, grid_w):
        """
        准备文本 tokens，正确处理视频占位符。

        Qwen2-VL 的 processor 需要知道视频尺寸来插入正确数量的视频 token。
        我们通过创建一个假的视频张量（只用于获取正确的 token 数量）来实现。
        """
        from prompt_template import build_prompt

        eval_dim = self.inferencer.data_config.eval_dim
        prompt_template_type = self.inferencer.data_config.prompt_template_type

        # 构造 chat_data
        chat_data = [[
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": build_prompt(prompt, eval_dim, prompt_template_type)},
                ],
            }
        ]]

        # 创建一个假的视频张量，只用于让 processor 计算正确的 token 数量
        # 尺寸: [T, C, H, W]，T = grid_t * 2, H = grid_h * 14, W = grid_w * 14
        dummy_video = torch.zeros(
            grid_t * 2,  # temporal_patch_size = 2
            3,
            grid_h * 14,  # patch_size = 14
            grid_w * 14,
            dtype=torch.float32
        )

        # 使用 processor 获取正确的 input_ids（包含正确数量的视频 token）
        batch = self.inferencer.processor(
            text=self.inferencer.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=None,
            videos=[dummy_video],
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": False},  # dummy video 已经是 [0,1] 范围
        )

        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device)
        }

    def compute_target_size(self, height, width, num_frames, max_pixels=None):
        """
        计算官方的目标 resize 尺寸。

        Args:
            height: 原始高度
            width: 原始宽度
            num_frames: 帧数
            max_pixels: 可选的最大像素数

        Returns:
            (target_height, target_width): 28 对齐的目标尺寸
        """
        from vision_process import smart_resize, FRAME_FACTOR

        VIDEO_MIN_PIXELS = 128 * 28 * 28
        VIDEO_MAX_PIXELS = 768 * 28 * 28
        VIDEO_TOTAL_PIXELS = 24576 * 28 * 28

        min_pixels = VIDEO_MIN_PIXELS
        total_pixels = VIDEO_TOTAL_PIXELS
        computed_max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / num_frames * FRAME_FACTOR), int(min_pixels * 1.05))
        final_max_pixels = max_pixels if max_pixels is not None else computed_max_pixels

        target_h, target_w = smart_resize(
            height, width,
            factor=28,
            min_pixels=min_pixels,
            max_pixels=final_max_pixels,
        )

        return target_h, target_w


import torch
import numpy as np

def verify_differentiable_path_logic(reward_model, video_path, prompt):
    """
    验证函数：对比官方路径（从文件读取）与手动路径（模拟 VAE 输出）的 Logits 差异。
    
    Args:
        reward_model: DifferentiableVideoReward 实例
        video_path: 视频文件路径
        prompt: 提示词
    """
    print(f"\n" + "="*20 + " 开始可微路径一致性验证 " + "="*20)
    device = reward_model.device
    num_frames = reward_model.inferencer.data_config.num_frames or 10
    max_pixels = reward_model.inferencer.data_config.max_frame_pixels
    
    # --- 1. 官方路径：获取参考 Logits ---
    # 我们直接调用内部的 inferencer.reward，但设 use_norm=False 以对比原始 Logits
    with torch.no_grad():
        # 获取官方处理后的 batch，用于获取准确的 target_size 和 input_ids 供后续对比
        batch_official = reward_model.inferencer.prepare_batch(
            [video_path], [prompt], num_frames=num_frames, max_pixels=max_pixels
        )
        # 运行官方推理
        outputs_off = reward_model.inferencer.model(return_dict=True, **batch_official)
        logits_official = outputs_off["logits"].float().cpu() # [1, 3]

    # 从官方 batch 中提取目标尺寸（由 Grid THW 反推）
    grid_thw = batch_official['video_grid_thw'][0]
    target_h, target_w = grid_thw[1].item() * 14, grid_thw[2].item() * 14
    print(f"[验证信息] 目标分辨率: {target_w}x{target_h}, 采样帧数: {num_frames}")

    # --- 2. 手动路径：模拟 VAE 输出并计算 ---
    # A. 首先以可微分的方式读取视频并 Resize（模拟 VAE 解码后的图像流）
    # 我们使用之前定义的 get_video_tensor_for_reward 确保采样逻辑一致
    video_tensor, _, _ = get_video_tensor_for_reward(
        video_path, 
        num_frames=num_frames, 
        max_pixels=max_pixels,
        sample_type=reward_model.inferencer.data_config.sample_type,
        device=device
    )
    
    # B. 关键步骤：将 [0, 255] 的像素张量转换为 VAE 风格的 [-1, 1] 范围
    # 这样才能真实模拟从 VAE 出来的输入
    vae_style_input = (video_tensor / 127.5) - 1.0
    vae_style_input = vae_style_input.detach().requires_grad_(True) 
    
    # C. 调用你要验证的函数
    # 注意：为了公平对比，这里的 target_height/width 必须使用从官方路径拿到的对齐后的尺寸
    logits_manual = reward_model.compute_reward_from_vae_output(
        vae_output=vae_style_input,
        prompt=prompt,
        target_height=target_h,
        target_width=target_w
    ).float().cpu()

    # --- 3. 差异计算 ---
    diff = torch.abs(logits_official - logits_manual)
    mse = torch.mean(diff**2).item()
    max_val = torch.max(diff).item()

    print("\n" + "-"*15 + " 对比结果 " + "-"*15)
    dims = ['VQ (视觉质量)', 'MQ (动作质量)', 'TA (图文相关)']
    for i, name in enumerate(dims):
        off_val = logits_official[0, i].item()
        man_val = logits_manual[0, i].item()
        print(f"{name:12}: 官方={off_val:8.4f} | 手动={man_val:8.4f} | 差值={diff[0, i].item():.6f}")

    print("-" * 40)
    print(f"Logits MSE: {mse:.10e}")
    print(f"最大绝对偏差: {max_val:.10e}")

    # --- 4. 梯度回传测试 ---
    print("\n[梯度测试]")
    loss = logits_manual.sum()
    loss.backward()
    if vae_style_input.grad is not None:
        grad_abs_mean = vae_style_input.grad.abs().mean().item()
        print(f"✅ 梯度回传成功! VAE 输入梯度均值: {grad_abs_mean:.10e}")
    else:
        print("❌ 梯度回传失败！请检查代码中是否存在断开梯度的操作（如 .item() 或 .numpy()）")

    if mse < 1e-4:
        print("\n✨ 结论：验证通过！手动构造的路径与官方路径完全一致。")
    else:
        print("\n⚠️ 结论：存在显著差异，请检查 Resize 插值方式或 Normalize 参数是否对齐。")

    return mse

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化
    reward_model = DifferentiableVideoReward("./checkpoints", device="cuda:0")

    # 2. 设置测试数据
    test_video = "/home/wubin/wanx-code/data_mixkit/data/video/mixkit-80s-man-dancing-to-radio-music-41321.mp4" # 换成你现有的路径
    test_prompt = "A high quality video with smooth motion."

    # 3. 执行验证
    verify_differentiable_path_logic(reward_model, test_video, test_prompt)

# if __name__ == "__main__":
#     # 使用修复后的 V2 验证函数
#     inferencer = VideoVLMRewardInference("./checkpoints", device="cuda:0")

#     video_dir = "/home/wubin/wanx-code/data_mixkit/data/video"
#     df_results = batch_validate_reward_consistency_v2(inferencer, video_dir, num_samples=100)

#     # 保存结果备查
#     df_results.to_csv("reward_consistency_test_v2.csv", index=False)
# # --- 执行验证 ---
# if __name__ == "__main__":
#     # 初始化
#     from inference import VideoVLMRewardInference
#     inferencer = VideoVLMRewardInference("./checkpoints", device="cuda:0")
    
#     video_path = "/home/wubin/wanx-code/data_mixkit/data/video/mixkit-a-couple-arguing-and-struggling-42244.mp4"
#     prompt = "The camera remains still, a couple is arguing and struggling in a room."

#     # 运行对比
#     compare_manual_vs_official_rewards(inferencer, video_path, prompt)


# # --- 测试代码 ---
# if __name__ == "__main__":
#     inferencer = VideoVLMRewardInference(load_from_pretrained, device=device)

#     video_path = "/home/wubin/wanx-code/data_mixkit/data/video/mixkit-a-couple-arguing-and-struggling-42244.mp4" # 找一个存在的视频
#     vae_output = get_mock_vae_output(video_path)
#     nframes, _, height, width = vae_output.shape
#     vae_output = (vae_output+1) / 2 * 255.0  # 转回 [0, 255] 范围，模拟原始视频帧

#     resized_height, resized_width = smart_resize(
#             height,
#             width,
#             factor=28,
#             min_pixels=100000,
#             max_pixels=200704,
#         )
#     video_differentiable = transforms.functional.resize(
#                 vae_output,
#                 [resized_height, resized_width],
#                 interpolation=InterpolationMode.BICUBIC,
#                 antialias=True,
#             )
#     print("Original VAE Output Shape:", vae_output.shape)
#     print("Resized Video Shape:", video_differentiable.shape)
#         # 4. 执行 9 维对齐的可微预处理
#     # 这步将 [T, C, H, W] 转换为 [Tokens, 1176]
#     pixel_values = differentiable_process_vlm_video_v446(video_differentiable, inferencer.processor)
    
#     # 5. 构造模型所需的 Batch
#     # 我们需要从官方获取 input_ids (文本部分是不需要梯度的)
#     prompt = "The camera remains still, a couple is arguing."
#     with torch.no_grad():
#         # 获取一个真实的 batch 用来拿 input_ids 和 grid_thw
#         batch_official = inferencer.prepare_batch([video_path], [prompt], num_frames=10)
    
#     # 构建最终用于反传的 Batch
#     # 注意：要把 pixel_values 替换为我们带梯度的版本，并转为模型对应的 dtype (bfloat16)
#     model_dtype = inferencer.model.dtype
#     input_batch = {
#         "input_ids": batch_official["input_ids"].to(inferencer.device),
#         "attention_mask": batch_official["attention_mask"].to(inferencer.device),
#         "video_grid_thw": batch_official["video_grid_thw"].to(inferencer.device),
#         "pixel_values_videos": pixel_values.to(model_dtype).to(inferencer.device)
#     }

#     # 6. 前向传播 (Reward 推理)
#     # 注意：此处不能用 torch.no_grad()，因为我们要留着梯度
#     print(">>> 正在进行可微前向传播...")
#     outputs = inferencer.model(return_dict=True, **input_batch)
#     logits = outputs["logits"] # 假设输出形状为 [1, 3] (VQ, MQ, TA)
    
#     # 计算一个标量奖励 (例如三个维度的总和)
#     reward_score = logits.sum()
#     print(f"Reward Score: {reward_score.item():.4f}")

#     # 7. 梯度回传验证
#     print(">>> 正在执行反向传播...")
#     reward_score.backward()

#     # 8. 检查最初的 vae_output 是否拿到了梯度
#     if vae_output.grad is not None:
#         grad_mean = vae_output.grad.abs().mean().item()
#         print(f"✅ 梯度回传成功!")
#         print(f"VAE Output Grad Mean: {grad_mean:.10e}")
        
#         # 简单检查梯度是否非零
#         if grad_mean > 0:
#             print("🚀 梯度信号有效，可以开始 RL 训练循环。")
#         else:
#             print("⚠️ 梯度为零，请检查模型是否处于 eval() 模式或是否有某些层阻断了梯度。")
#     else:
#         print("❌ 梯度回传失败，vae_output.grad 为 None。")

#     # 9. (可选) 清理显存
#     del outputs, input_batch, pixel_values
#     torch.cuda.empty_cache()
    


# if __name__ == "__main__":
#     # 初始化你的推理类
#     # 注意：确保 checkpoints 路径正确
#     load_from_pretrained = "./checkpoints" 
#     device = "cuda:0"
#     inferencer = VideoVLMRewardInference(load_from_pretrained, device=device)

#     video_path = "/home/wubin/wanx-code/data_mixkit/data/video/mixkit-a-couple-arguing-and-struggling-42244.mp4" # 找一个存在的视频
#     prompt = "The video begins in a bowling alley setting, showcasing a bowling ball return machine filled with colorful balls - purple, yellow, blue, red, and green. Initially, only the person's legs and feet, clad in tan pants and red shoes, are visible, suggesting they are standing or walking in the alley. The background is adorned with the typical wooden lane surface and purple and pink lights, contributing to the alley's ambiance.\n\nAs the video progresses, the person's hands come into view, reaching into the bowling ball return machine. One hand grasps the yellow bowling ball, while the other appears to be steadying or adjusting the purple ball. The red and green balls remain undisturbed inside the machine. The person seems to be in the process of retrieving or organizing the bowling balls from the machine, with the background elements, including the wooden lane surface and colored lights, remaining consistent throughout this sequence.",


#     with torch.set_grad_enabled(True):
#         verify_v446_consistency(inferencer, video_path, prompt)

# if __name__ == "__main__":
#     load_from_pretrained = "./checkpoints"
#     device = "cuda:0"
#     dtype = torch.bfloat16

#     inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)

#     video_paths = [
#         "/home/wubin/wanx-code/data_mixkit/data/video/mixkit-a-person-takes-a-bowling-ball-and-makes-a-shot-49102.mp4",
       
#     ]

#     prompts = [
#         "The video begins in a bowling alley setting, showcasing a bowling ball return machine filled with colorful balls - purple, yellow, blue, red, and green. Initially, only the person's legs and feet, clad in tan pants and red shoes, are visible, suggesting they are standing or walking in the alley. The background is adorned with the typical wooden lane surface and purple and pink lights, contributing to the alley's ambiance.\n\nAs the video progresses, the person's hands come into view, reaching into the bowling ball return machine. One hand grasps the yellow bowling ball, while the other appears to be steadying or adjusting the purple ball. The red and green balls remain undisturbed inside the machine. The person seems to be in the process of retrieving or organizing the bowling balls from the machine, with the background elements, including the wooden lane surface and colored lights, remaining consistent throughout this sequence.",
#    ]

#     with torch.no_grad():
#         rewards = inferencer.reward(video_paths, prompts, use_norm=True)
#         print(rewards)
