# ClipRank 固定输入扩大评测

2026-10-22：独立选择 10 个未使用提示词，每组前 2 段视频，共 20 段，比较 4 帧 224/448 像素，共 40 次评分。模型、生成参数和选片权重固定，先提交清单后推理，不用本轮结果重新挑参数。

数据和方案：reports/expanded-evaluation-plan。运行结果：D:/career-projects/runs/ClipRank-expanded/20261022_094212。专用环境：D:/career-projects/envs/videoscore2/Scripts/python.exe（videoscore2-cuda）。

## Windows VS Code 操作

在 Remote SSH 工作区的“运行和调试”中选择 ClipRank | expanded evaluation | resume。

- 若实验仍在运行，进程锁会拒绝重复启动；不会再次加载模型占用显存。
- 若之前进程意外结束，完成的评分会直接读取，只继续未完成任务。
- 若清单、模型版本、提示规则、评测代码或选片规则改变，则拒绝续跑，必须建立新实验。
- 原始模型输出和每次尝试均保存。失败或截断输出不会自动重跑以挑选更好结果。

本轮已安排 finish_expanded_evaluation.py 接续处理：等待完成 → 校验 → 导出报告 → 提交并推送。若实验异常退出、工作区有其他未提交修改或网络推送失败，流程保留状态并停止，不覆盖或重复推理。

后台状态与日志：D:/career-projects/videoscore2-setup-20261022/expanded-evaluation.log、delivery-state.json、delivery.log。

完成后报告导出至 reports/expanded-evaluation/RESULTS.md。记录三维 MAE、候选排序、Spearman/Pearson、覆盖率、按提示词 bootstrap 区间及推理代价。20 段仍是诊断子集，不能当作完整基准或线上效果。元数据已固定版本，外部对象存储视频用下载后的 SHA256 标识实际字节。

## 双视频比较开发实验

先复核 4 组未判对候选的采样帧和解释，见 reports/error-review/AUDIT.zh-CN.md。新模块 application/pairwise.py 同时比较两段视频，并交换 AB/BA 顺序；每维一致才给出选择，其余进入人工复核。

run_pairwise_development.py 仅在这 4 组已知错误上运行 8 次双视频推理，用于验证格式、显存与顺序一致性。该组由历史错误筛出，不作为独立检验或整体效果提升依据；也不会自动替换当前应用默认评分。第一组两次输出均无效时提前停止并保留原始回答。

本轮结果保存在 D:/career-projects/runs/ClipRank-pairwise-development；完成后由 finish_pairwise_development.py 保存 reports/pairwise-development/RESULTS.md 并尝试提交推送。后续需单独做已正确样例回归和未使用提示词检验。
