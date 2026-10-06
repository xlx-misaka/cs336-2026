# Experiment Log

每次运行 `src/main.py` 都会在 `experiments/<experiment_name>/<timestamp>/` 下生成：

- `config.json`：本次实验的完整超参数。
- `metrics.csv`：按 `step` 和 `elapsed_seconds` 记录的训练/验证指标。
- `checkpoints/`：周期性保存的模型 checkpoint。

## 日志字段

| 字段 | 含义 |
| --- | --- |
| `step` | 已完成的梯度更新步数 |
| `elapsed_seconds` | 从本次训练开始经过的墙钟时间 |
| `train_loss` | 当前 mini-batch 的训练 loss |
| `val_loss` | 在验证集上估计的 loss |
| `lr` | 当前学习率 |
| `tokens_seen` | 已处理的 token 数 |

## 实验记录

| run id | experiment | learning rate | warmup | batch size | steps | device | final train loss | final val loss | diverged | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| | | | | | | | | | | |
