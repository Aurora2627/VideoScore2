# ClipRank 冻结多模态骨干的评分头训练实验

训练/验证数据来自 VideoFeedback2 官方训练划分；新测试子集来自官方测试划分，按提示词隔离，并排除先前项目调试使用的组。
本轮训练轻量评分头，未更新 VideoScore2 骨干，未执行 LoRA、SFT 或 GRPO。骨干此前已在官方训练集上训练过，因此验证集隔离针对本项目评分头，不代表骨干从未见过验证样本。

## 数据与训练

样本数量：{"train": 755, "validation": 188, "test": 63}。
特征：四帧、448 输入、NF4 量化 VideoScore2，读取回答生成前最后一个非 padding token 的隐藏状态。模型输入不含人工评分或解释。
线性头、MLP、MLP+排序损失各三个随机种子；仅按验证集 MAE 选择三种子集成。选中：mlp。

## 新测试子集结果

| 方法 | 样本数 | 平均 MAE | 排序正确/可比较对 |
|---|---:|---:|---:|
| 评分头（全部测试） | 63 | 0.69112 | 6/13 |
| 训练集均值 | 63 | 0.896191 | 0/13 |
| 固定三分 | 63 | 0.857143 | 0/13 |
| 原模型（有效输出） | 63 | 0.68254 | 7/13 |
| 评分头（同一有效子集） | 63 | 0.69112 | 6/13 |

原模型有效输出：63/63；无效样本不计作正确，配对比较仅使用同一有效子集。
按提示词 bootstrap：{"estimand": "Prompt-macro mean MAE difference: head minus original; negative favors head", "groups": 48, "observed": -0.008412972092628463, "ci95": [-0.10267397190133729, 0.09111462362731498], "bootstrap_samples": 5000, "seed": 20261022}。

测试只在验证集选择完成后运行；这批结果不能用于继续调参后再次声称独立验证。测试规模有限，报告保留退步与不确定性，不自动替换当前应用默认评分器。
评分头、标准化参数、九次训练曲线、验证集对照、测试原始回答、采样和文件校验均保存。视频和完整骨干权重不提交 Git。

基础模型与数据来源：[VideoScore2](https://huggingface.co/TIGER-Lab/VideoScore2)、[VideoFeedback2](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2)。代码与模型授权见项目 LICENSE、MODEL_SOURCES.md 和 CITATION.cff。
