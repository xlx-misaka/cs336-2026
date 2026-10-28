# CS336: Language Modeling from Scratch — Spring 2026

Stanford CS336「从头构建大语言模型」2026 春季课程资料与作业实现。
讲师：Tatsunori Hashimoto、Percy Liang。课程安排与录像见[官网](https://stanford-cs336.github.io/)。

## 作业进度

| 作业 | 内容 | 状态 |
| --- | --- | --- |
| [1 · Basics](assignments/assignment1-basics/) | BPE、Transformer、训练循环 | 已完成 |
| [2 · Systems](assignments/assignment2-systems/) | Triton、Flash Attention、分布式训练 | 基本完成，待完整实验验证 |
| [3 · Scaling](assignments/assignment3-scaling/) | Scaling laws 实验与拟合 | 未开始 |
| [4 · Data](assignments/assignment4-data/) | 数据过滤、去重、质量分类 | 未开始 |
| [5 · Alignment](assignments/assignment5-alignment/) | SFT、RLHF、RLVR | 未开始 |

## 资料导航

- [lectures/](lectures/)：17 讲课程材料，包含 PDF 幻灯片与可运行 Python 讲义。
- [assignments/](assignments/)：5 次作业的 handout、代码与测试，使用说明见各作业 README。
- [readings/papers/](readings/papers/)：阅读论文 PDF，完整引用见 [references.py](lectures/references.py)。

Python 讲义可直接阅读源码，运行时使用 `lectures/` 中的 `uv` 环境；[在线讲义](https://stanford-cs336.github.io/lectures/?trace=lecture_01)也可直接浏览。

数据集、虚拟环境、训练 checkpoint 和缓存不纳入版本控制，复现方式见各作业 README。

*资料来自 Stanford CS336 官网及官方仓库，仅供个人学习，版权归 Stanford 及各作者所有。*
