# ClipRank 评分头训练

在已有 VideoScore2 推理评估项目中增加冻结骨干的监督训练。此阶段训练评分头，不更新多模态骨干，不属于 LoRA/SFT/GRPO。

## 固定设计

- Windows、videoscore2-cuda 虚拟环境、PyTorch CUDA、RTX 4060 8GB。
- 固定基础模型版本 09a2732cb64fa566a1f332f978368292ce5c295c，NF4 双重量化、BF16、SDPA。
- 每视频均匀采样四帧，输入最大边 448，与当前原模型基线保持一致。
- 不生成回答，读取骨干最后一个有效 token 的多模态隐藏向量，缓存后训练评分头。输入仅含视频、任务定义和生成提示词；评分和解释只作为监督/评估数据。
- 官方训练集筛出 128 个训练提示词组（755 段）、32 个验证提示词组（188 段）；官方测试集选取尚未用于本项目调试的 48 个提示词组（63 段）。不按标注分数筛样本。优先选取有多个候选的测试提示词，便于测量排序，因此测试子集并非官方 500 段全量基准的无偏估计。
- 训练/验证均排除整个官方测试集中的提示词 ID 和规范化文本；跨集合还检查视频 SHA256。对语义近似而文字不同的提示词尚未做完整语义去重。
- 原骨干已在 VideoFeedback2 官方训练集上训练过；本项目验证集独立于新评分头拟合，不代表对骨干完全未见。不得声称从零训练或完成了完整基础模型微调。

## 训练与选择

线性头、64 隐藏单元 MLP、MLP+排序损失三个变体，各训练种子 42/43/44。标准化统计只从训练特征拟合。输出为 1+4*sigmoid，三维连续分数并非概率。

回归采用 SmoothL1(beta=0.5)。排序变体额外加入 0.1 倍 pairwise logistic loss，仅比较同一规范化提示词内标注不同的候选。AdamW，学习率 0.001，weight_decay=0.01；每批八个提示词组，最多 100 epoch，验证集 MAE 连续 15 epoch 不改善则停止，梯度裁剪 1.0。

每个种子保留最佳验证 MAE 的 checkpoint，再按三个种子的集成验证 MAE 选变体。新测试集只在选择之后评估一次；比较原模型、训练集均值和固定三分，报告 MAE、相关性、候选排序和按提示词 bootstrap 区间。保留退步，不自动替换默认应用。

## 运行与调试

打开同目录的 ClipRank-training.code-workspace，可在 VS Code 中选择 videoscore2-cuda 解释器并调试数据划分、特征提取和评分头训练。各 Python 入口支持 --help。环境依赖沿用已安装版本，无需改动其他项目环境。

1. inspect_dataset.py 下载固定版本 RL 评分标注，记录来源和 SHA256。
2. prepare_data.py --download 按固定计划下载数据，保存清单和逐视频校验值。
3. run_pipeline.py --run-dir <新目录> --publish 顺序运行特征提取、九次评分头训练、独立测试与原模型对照，完成后导出 reports/head-training 并提交推送。
4. predict.py --video <视频路径> --prompt <生成提示词> --checkpoint <selected-ensemble.pt路径> --output <结果路径> 使用训练后的头给新视频评分。

运行目录保存所有特征张量、训练 checkpoint、曲线、原始测试回答和日志。Git 中保存源码、划分、模型头及实验报告，不上传视频和完整骨干。流水线首次运行要求新目录；异常时记录 needs_attention，不自动重新调参或重复启动。特征提取支持同一清单与源码版本下复用缓存。

## 来源

基础模型：[VideoScore2](https://huggingface.co/TIGER-Lab/VideoScore2)。数据：[VideoFeedback2](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2)，固定版本 99b9935b00a205756c25b0555f2dcdc071281327。仅使用评分，不使用数据集中生成的推理解释作为输入。部分训练视频地址由官方测试元数据验证的目录/生成器规则导出，下载字节用 SHA256 标识；无法唯一映射的生成器后缀被明确排除，此子集不代表全部训练分布。
