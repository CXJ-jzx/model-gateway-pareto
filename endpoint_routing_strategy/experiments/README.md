# 路由实验

所有实验配置、入口和输出集中在这里，不改变正式策略算法。

交接基线和完成度见 [项目交接指南](../README.md)。后续 SLO 小实验计划在工作区根目录 `slo_experiments/` 独立开展，不覆盖此处的数据准备配置、选择场景或既有结果；目前尚未创建。

| 目录 | 作用 |
| --- | --- |
| `configs/` | 数据准备和选择场景配置 |
| `selection/` | 当前请求感知策略、规则验证、开销测量、最小失败切换 |
| `busy_check.py` | 合成用量快照上的轻量繁忙判断和阶段开销记录 |
| `legacy/` | 本地历史工具，不随 GitHub 发布 |
| `output/` | 运行自动生成的数据/结果，不随 GitHub 发布；本地已有索引为 `output/README.md` |
| `archive/` | 本地旧产物归档，不随 GitHub 发布 |

## 当前选择与失败切换

项目根目录执行：

```powershell
python -B -m endpoint_routing_strategy.experiments.selection --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_20261003 --benchmark-repeats 30
python -B -m endpoint_routing_strategy.experiments.selection.failover_demo --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_20261003
```

不传数据目录可用完全合成请求。每次自动生成新时间戳目录，不覆盖旧报告。配置默认覆盖三个模型两种输出模式，实际场景数量以配置和本次摘要为准。

`selection` 调用正式 `RoutingEngine.route_request()`，验证硬约束、预测费用、二维 Pareto、首选评分与稳定备用。失败演示只注入少量 503/200 Mock 结果，验证切换与反馈，不发送 HTTP、不建设复杂 Endpoint 模拟器。

传 `--dataset-dir` 仅用于到达元数据和模型身份模板，Token 预测、SLO、优先级和 Endpoint 状态仍为显式场景输入，不是基于原流量的真实容量/结果回放。

每个场景保存输入快照、`decision.json`、候选 CSV、Pareto SVG 和阶段开销；总目录保存配置、来源哈希、`summary.json` 和 `summary.md`。没有价格或必要指标的点只有检查原因，不伪造图上坐标。

详细公式及文件/函数流程见 [selection/README.md](selection/README.md)。

## 繁忙判断

```powershell
python -B -m endpoint_routing_strategy.experiments.busy_check
python -B -m endpoint_routing_strategy.experiments.busy_check --enter-threshold 0.9 --exit-threshold 0.7
```

每次生成新的 `output/busy_check_时间戳/report.json`：保存输入用量、局部和模型判断、验证结果、计算耗时与源代码哈希。只验证容量压力，不读取完整流量、不模拟 Endpoint、不改选路评分，不证明时延改善。规则与边界见 [繁忙判断说明](../docs/繁忙判断说明.md)。

## 数据准备与独立校验

```powershell
python -B -m endpoint_routing_strategy.data prepare --traffic .\实验数据\历史性能数据包\流量记录_历史性能.jsonl
python -B -m endpoint_routing_strategy.data validate --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_20261003
```

`incoming_requests.jsonl` 是到达时元数据，`requests.jsonl` 是补充画像后的调度输入。完成事实在 `canonical_requests.jsonl`、`observations.jsonl`，不直接混入当前请求的选路输入。说明见 [data/README.md](../data/README.md)。

## 历史入口与结果管理

旧 route/sweep/replay 已移入本地 `legacy/`，功能保留但不上传，费用/SLO口径不同，不作为正式策略入口。

旧生成产物压缩归档前核对全部文件内容；展开旧目录清理后可从压缩包恢复。原始日志、当前输入数据、源代码和设计资料不通过这种结果清理删除。未来生成的新目录不会被自动清理；需要时再按相同原则选择保留和归档。
