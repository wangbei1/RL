# Reward Model 全梯度回传（full gradient）显存优化建议

## 1) 最近提交脉络（与 RL 显存相关）

- `8762673`：引入了「全梯度路径」下的显存优化：
  - Reward model gradient checkpointing
  - VAE 时序 chunk 解码 + checkpoint
  - Reward model FSDP 分片
- `4d091cc`：将 RL loss 重构为 REINFORCE（高斯扰动 + 反向差分），**不再穿过 reward model 与 VAE 反传**，以换取更低显存。

结论：从提交历史看，项目已经从“全梯度优化显存”演进到“避免全梯度路径（REINFORCE）”以进一步降显存。

## 2) 如果必须保留“梯度完整穿过 reward model”，优先级最高的降显存手段

> 下列建议针对“reward model 参数冻结，但计算图必须可导到 generator”场景。

### A. 激活检查点（checkpointing）覆盖 VAE + Reward Model（优先级 S）

1. Reward model 开启 `gradient_checkpointing_enable(use_reentrant=False)`。
2. VAE decode 使用时序 chunk（如 `vae_chunk_size=3~6`）并对每 chunk 做 checkpoint。

- 典型收益：显存大幅下降（常见 25%~45%，取决于 T、分辨率、模型层数）。
- 代价：吞吐下降（通常 20%~60%）。

### B. 降低 reward 输入尺寸（优先级 S）

- 直接下调 `rl_target_height/rl_target_width`，例如从 `336x504` 降到 `280x420` 或 `252x378`（保持 28 的倍数）。
- 这是全链路激活与 attention 复杂度同步下降，通常是最“线性有效”的手段。

### C. Reward model 参数分片（FSDP / ZeRO-3）（优先级 A）

- 开启 `rl_reward_fsdp=true`（多卡时）。
- 可显著减少单卡参数常驻显存，尤其 reward backbone 较大时更明显。

### D. 更激进的 mixed precision（优先级 A）

- 统一 BF16（推荐）并确保 autocast 覆盖 reward forward。
- 若硬件支持并且精度可接受，可对 reward model 试 FP8/INT8 权重量化（只做前向，梯度对输入仍可保留）。

### E. 微批与梯度累积（优先级 A）

- 把 `batch_size` 压到 1，靠 `gradient_accumulation_steps` 维持等效 batch。
- 在视频 RL 下，这往往是“最稳”的兜底方案。

### F. 时序分块反传（Truncated temporal backprop）（优先级 B）

- 将完整视频切为窗口（如 21 帧切成 3x7 或 4x5+1），分窗口计算 reward 并累积。
- 若 reward 对全局时序依赖较强，会带来轻微目标偏差，但显存收益可观。

## 3) 建议的落地顺序（从低风险到高收益）

1. 先开 checkpoint（reward + VAE chunk）。
2. 再降 reward 输入分辨率。
3. 再启用 reward FSDP 分片。
4. 最后做量化、时序截断等“有分布偏差风险”的优化。

## 4) 可直接尝试的一组配置（full-gradient 模式）

```yaml
# 目标：在保持 full-gradient 的前提下显著降显存
mixed_precision: true
batch_size: 1

# reward 输入降采样
rl_target_height: 280
rl_target_width: 420

# reward 参数分片（多卡）
rl_reward_fsdp: true

# VAE 时序 chunk（建议从 5 降到 3~4 试）
vae_chunk_size: 4
```

并配合：
- reward model `gradient_checkpointing_enable(use_reentrant=False)`
- 训练端启用 gradient accumulation

## 5) 经验判断：哪些手段“最显著”

若你只想要“显著下降”，通常组合是：

- **`checkpointing + 分辨率下调 + batch=1 + 累积`**（单机最有效）
- 多卡再加 **reward FSDP**（进一步降单卡峰值）

如果你不强制 full-gradient，当前 `REINFORCE` 路线一般会比上述 full-gradient 方案再省一大截显存。
