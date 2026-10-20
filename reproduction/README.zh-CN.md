# VideoScore2 项目说明

本项目复现 VideoScore2 的视频质量评估推理，并在 Windows RTX 4060 Laptop 8GB 上做低显存适配。

输入是生成视频和生成时使用的提示词，输出是视觉质量、文本对齐、物理与常识一致性三个维度的 1–5 分及分析。它可以帮助检查生成视频的缺陷；目前没有接入视频生成服务或训练闭环。

## 当前实现

流程：读取真实视频 → 均匀抽取 4 帧 → 缩小最长边至 224 像素 → 官方多模态处理器编码视频和文本 → 官方 VideoScore2 权重进行 CUDA 推理 → 检查生成是否到达 EOS → 提取最终三项评分 → 与标签对比并保存证据。

使用 Python、PyTorch、Transformers、OpenCV、Accelerate、bitsandbytes。模型采用 NF4 4-bit 双重量化、BF16 计算和 SDPA 注意力；使用贪心解码，生成上限为 1024 token。

## 环境与 VS Code

环境显示名为 videoscore2-cuda，实际目录仍为 D:/career-projects/envs/videoscore2。显示名与目录名可以不同。保留原目录可以避免已安装工具中的绝对路径失效。

打开仓库内 VideoScore2.code-workspace，连接到 Windows 后新建终端，终端名称应为 VideoScore2 CUDA，提示符应显示 (videoscore2-cuda)。该终端不加载用户 PowerShell 配置，以避免自动进入 Conda base；不修改全局 Conda 设置。

也可在 PowerShell 中运行 . ./reproduction/activate.ps1。其他电脑可通过 -EnvironmentPath 指定已创建的环境路径。激活只影响当前终端。解释器和调试配置均使用项目独立环境。

运行 python reproduction/check_environment.py --output 一个新的JSON文件路径，可以检查环境身份和小型 CUDA 运算。不要覆盖已有报告。模型调试选择 VideoScore2 | videoscore2-cuda | inference，提供视频路径、原始提示词和一个尚不存在的结果目录。

## 已有结果与边界

已跑完固定的 3 个真实视频。预测与标签（依次为视觉质量/文本对齐/物理一致性）：

| 视频 | 预测 | 标签 |
|---|---|---|
| 004882_h | 3/2/3 | 4/2/1 |
| 001033_p | 3/3/3 | 3/2/3 |
| 002314_j | 1/3/2 | 1/1/1 |

生成约耗时 141、105、99 秒，峰值已分配显存约 5.7 GiB，均不包含模型加载时间。完整记录见 reports/diagnostic-three。首样本只修复了解析格式，原始失败记录保留，并有单独的重新解析记录。

三个视频都完整输出，不代表三个预测全部正确。当前存在评分偏差，不能从三个样本推断总体效果。量化、帧数和分辨率均与完整论文评测有差异；尚未完成全量基准评测或 SFT/GRPO 训练。

## 哪些是复现工作

原作者贡献模型、数据、训练方法和论文结果。本仓库新增工作是独立环境配置、8GB 显卡推理适配、输入校验、输出解析与异常处理、真实样本诊断和可追溯实验记录。

关键源码：infer_cuda.py 为单视频推理；run_diagnostic.py 为固定样本诊断；reparse_saved.py 在不重跑模型的情况下重新解析输出；summarize_diagnostic.py 校验并汇总记录。原始推理、源码快照和 Git 历史均保留。

后续适合学习和改进的内容：视频采样与时序信息、视觉 token 和显存关系、量化误差、评分协议，以及在独立验证集上比较不同帧数与分辨率的质量和成本。扩大评测后再判断模型效果。
