<<<<<<< HEAD

# Model Gateway Pareto：Endpoint 路由策略

本项目在**模型已经由上游确定**的前提下，研究如何选择提供该模型的 Endpoint。当前是可审计的 Python 路由原型和离线规则验证工程，不是已经接通真实服务的完整网关。

包版本 `0.3.0`。仓库只发布正式代码、测试、必要配置和当前说明，不包含原始流量、实验输出、旧实验、归档或调研资料。核心路由算法不因发布而改变，新的 SLO 实验尚未启动。

## 交接人先看什么

1. 阅读 [正式项目 README](endpoint_routing_strategy/README.md)：完成度、数据流程、关键函数、公式、执行指令、修改位置及待办。
2. 阅读 [实验说明](endpoint_routing_strategy/experiments/README.md)：用纯合成输入复现选路与失败切换，不依赖未上传的数据。
3. 阅读 [数据说明](endpoint_routing_strategy/data/README.md) 和 [选择与失败切换说明](endpoint_routing_strategy/experiments/selection/README.md)：需要深入时再查。
4. 查看 [状态总结](endpoint_routing_strategy/docs/状态维护总结.md)：内存结构、字段和更新来源。

## 工作区目录

GitHub checkout 的最小结构：

```text
README.md
.gitignore
pyproject.toml
endpoint_routing_strategy/
  *.py               正式内存状态、价格、路由、失败反馈和压力判断
  requirements.txt   运行依赖记录（目前标准库）
  data/              数据清洗、画像、准备与独立验证
  experiments/       配置、当前选择实验、失败演示和压力验证
  docs/              当前设计说明及后续路线图
  tests/             回归测试
```

以下是原开发工作区的其他目录说明，**它们不是仓库自带内容**：

| 路径                                                     | 用途与维护边界                                           |
| ------------------------------------------------------ | ------------------------------------------------- |
| `endpoint_routing_strategy/`                           | 正式 Python 包，包含数据处理、内存状态、策略、实验、测试和设计文档             |
| `实验数据/`                                                | 当前原始数据位置；含流量记录、端点配置、历史性能数据包及独立 prompt/回答数据；只读，不覆盖 |
| `figures/`                                             | 用户维护的架构/流程图及导出文件；图中规划不代表代码已实现                     |
| `archives/`                                            | 本次交接基线 ZIP、逐文件哈希清单和恢复说明                           |
| `资料/`、`实验数据1/`、`流量分析实验（废）/`                            | 已有调研或其他实验资料，不是本项目的正式运行入口，未自动删除                    |
| `Endpoint调度策略流程图.*`、`调研*.md`、`prompt数据.zip`、`性能数据.zip` | 已有资料与导出文件；保留原位置                                   |
| `pyproject.toml`、`.gitignore`                          | 包配置、Python 版本约束及生成文件忽略规则                          |

旧文档、CLI 默认值或历史 manifest 中的 `历史性能数据包/` 是此前位置；当前实际位置为 `实验数据/历史性能数据包/`。不要为了兼容旧路径而复制数据或修改历史来源记录，运行数据处理时显式传 `--traffic`。

## 最短运行路径

在 `E:\Desktop\model_gateway`、Python 3.11+ 环境执行：

```powershell
python -B -m unittest discover -s .\endpoint_routing_strategy\tests -v
python -B -m endpoint_routing_strategy.experiments.selection --benchmark-repeats 30
python -B -m endpoint_routing_strategy.experiments.selection.failover_demo
```

后两条不发 HTTP，自动生成独立结果目录。完整的数据准备、复验和已有数据入口见 [正式项目 README](endpoint_routing_strategy/README.md#2-运行与复验)。

原工作区交接基线为 185 项测试、183 通过、2 跳过；发布副本不含旧实验和原始数据，预计 181 通过、4 跳过。跳过项只涉及两个旧入口和两个原始数据价格案例，正式路由、数据契约及合成实验仍可执行。测试命令会给出实际数量。

## 当前结论与下一阶段

已经完成：初始请求元数据重建、防结果泄漏检查、内存索引与窗口、预测费用和价格档、硬容量/SLO 检查、Pareto/线性评分、稳定备用、独立容量压力判断及可复现实验产物。

尚未完成：真实 Token/时延预测模型、真实 HTTP 执行及负载计数闭环、原子容量预占、延迟队列和重请求调度、数据库实现、其他策略对比、生产 SLO 达标验证。



下一阶段单独开展 **SLO 小实验**，建议新建根目录 `slo_experiments/`，与当前包并列；实验目录不等于 Git 分支。先做条件性能统计和独立验证，再决定是否接入 `RequestSLO`，不覆盖当前配置、输入、报告或核心代码。新增目录发布时须显式调整 `.gitignore` 的根目录白名单。具体边界见 [后续修改指南](endpoint_routing_strategy/README.md#8-后续如何修改)。
