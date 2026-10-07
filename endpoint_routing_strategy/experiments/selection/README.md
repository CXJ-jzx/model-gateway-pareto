# 本阶段：验证我们的 Endpoint 选择策略

本实验调用项目自己的 `RoutingEngine.route_request()` 与失败切换接口，不接入随机、轮询、最低价等其他策略，也不搭建复杂的 Endpoint 执行模拟器。目标是验证：给定初始请求和可解释的状态快照，程序是否按“可用性与预测容量过滤 → 请求费用与性能估计 → 逐项 SLO 过滤 → 二维 Pareto 选首选 → 稳定性优先选备用 → 失败后重新检查”的设计选择 Endpoint。

这些结果只能证明策略实现与规则一致，不能证明真实请求一定满足 SLO，也不能证明相较其他策略更省钱或延迟更低。真实效果评估需要后续接入执行结果与对照策略。

## 1. 如何运行

在项目根目录 `E:\Desktop\model_gateway` 打开 PowerShell。

使用已经准备好的初始请求数据作为模板：

```powershell
python -m endpoint_routing_strategy.experiments.selection --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_20261003 --benchmark-repeats 30
```

不提供数据目录时，运行完全合成的请求与状态快照：

```powershell
python -m endpoint_routing_strategy.experiments.selection --benchmark-repeats 30
```

指定修改后的场景配置和独立输出目录：

```powershell
python -m endpoint_routing_strategy.experiments.selection --config .\endpoint_routing_strategy\experiments\configs\selection_scenarios.json --output-dir .\endpoint_routing_strategy\experiments\output\selection_custom --benchmark-repeats 30
```

默认输出到 `experiments/output/selection_时间戳/`，各次运行分开存放。修改实验时建议使用新的输出目录，避免混淆不同配置的结果。

运行请求路由的回归测试：

```powershell
python -m unittest endpoint_routing_strategy.tests.test_request_routing -v
```

运行项目所有测试：

```powershell
python -m unittest discover -s endpoint_routing_strategy/tests -v
```

运行小型失败切换演示（首选失败，切换到稳定备用，不调用真实 HTTP）：

```powershell
python -m endpoint_routing_strategy.experiments.selection.failover_demo
```

默认输出到 `experiments/output/failover_时间戳/`。其中保存初始/反馈后状态、首选与重试两张 Pareto 图、完整会话、摘要和来源哈希。切换演示只报告少量 Mock 尝试结果，不建设 Endpoint 执行环境。

## 2. 输入是什么，不是什么

`--dataset-dir` 读取数据准备阶段的 `requests.jsonl`，通过 `RoutingRequest.from_prepared()` 转为初始请求。它不是原始的“请求完成日志”，也不会把 `final_endpoint_id`、实际 Token 消耗、最终延迟等结果作为本次选择的输入。

场景加载函数 `load_cases()` 使用真实数据中的请求标识、模型与到达时间作为模板，再依据场景配置明确修改预测 Token、SLO、优先级等字段。Endpoint 价格、容量、负载、健康状态、近期样本与先验都来自显式配置的合成快照，不是从真实配置推断出来的 Endpoint 环境，也不是某个未选择 Endpoint 的反事实执行结果。

默认场景覆盖三个模型及流式、非流式两种模式。提供的数据若缺少某个模型的一种模式，加载器会优先换用两种模式均有初始请求样本的其他模型；替换信息记录在 `provenance.model_mapping`、`model_substitution` 等字段中。如果没有足够的独立模型可替换，会跳过相关场景并记录 `skipped_case_ids`；如果完全没有双模式模型，会明确报错，不会偷偷补成“真实数据”。这里要求的是存在模板样本，不代表已经具备真实性能评估所需的统计样本量。

## 3. 正式路由逻辑

### 3.1 按模型查候选，检查可用性与预测容量

`model_id` 通过内存索引直接找到可提供服务的 Endpoint，不遍历匹配其他模型。禁用、冷却尚未结束、Endpoint 整体不健康、没有所需币种的有效价格，都会记录排除原因。

容量和当前用量均按 `(model_id, endpoint_id)` 维护，Endpoint 聚合用量不替代模型容量判断。接收请求后的预测值为：

```text
projected_rpm         = current_rpm + 1
projected_tpm         = current_tpm + predicted_input_tokens + predicted_output_tokens
projected_concurrency = current_concurrency + 1
```

预测值等于上限仍可接收，超过上限才排除。限制为 `None` 表示没有已知的配置限制，并不保证无限容量；默认 `unknown_capacity_policy="allow"` 放行未知容量，也可以显式设为 `"block"` 排除；限制为 `0` 表示不可用。TPM 当前采用“预测输入 + 预测输出总 Token 预留”的明确实验口径，后续若供应商的计费或限流口径不同，应修改该口径而不是混用。

本入口只计算是否可选，不预占用容量。一次决策不会给当前 RPM、TPM 或并发数加值；并发请求的原子预留与执行完成后的释放属于后续网关接入工作。

### 3.2 计算近期性能与历史先验融合

近期窗口按 `(model_id, endpoint_id, stream_type)` 隔离。只允许 `occurred_at <= cutoff` 的已完成样本进入决策，不读取未来结果。

`compute_candidate()` 从候选窗口中选择满足目标样本数的最短窗口，若均不足则使用最大窗口。窗口内样本按年龄赋予指数衰减权重：

```text
w = 2 ^ (-样本年龄分钟 / half_life_minutes)
```

一倍半衰期之前的样本权重为 `0.5`，两倍之前为 `0.25`。TTFT、TPOT、E2E 使用成功请求中具有对应指标的样本，计算加权 P95；成功率使用全部尝试的成功/失败标记做加权平均。失败样本不会被当作快速响应来改善延迟指标。

当前实现是“指数时间衰减的加权统计”，不是递推式 EWMA，也不是把窗口中的延迟简单求算术平均。加权 P95 代表近期慢尾部的统计特征，不是对本次请求实际延迟的保证。

各项指标分别计算有效样本数，再与先验融合：

```text
n_eff       = (sum(w)) ^ 2 / sum(w ^ 2)
prior_weight = prior_strength / (n_eff + prior_strength)
estimated_metric = (1 - prior_weight) * online_p95 + prior_weight * historical_prior
```

只有近期指标时用近期值，只有先验时用先验，二者都缺失则不能提供该指标。`prior_strength` 是可配置的先验强度，不是自动把历史快照的样本数量当作权重。候选记录同时保留原始在线 P95、融合估计与有效样本信息，便于追溯。

### 3.3 按每个有效 SLO 逐项过滤

非流式请求必须提供 E2E SLO；流式请求必须同时提供 TTFT 与 TPOT SLO。如果流式请求额外提供了 E2E SLO，也会检查 E2E。

判断使用上一步的逐指标融合估计，不使用单一的加权性能值代替约束。因此，即使 `η = 1`、评分只重视 TTFT，TPOT 超标仍会排除；`η = 0` 时 TTFT 约束也不会失效。

如果路由发生在请求到达之后，已经等待的时间从 E2E 和 TTFT 预算中扣除，TPOT 是单位 Token 的生成时间预算，不扣等待时间。剩余预算耗尽时不会给请求重新发放一份完整预算。硬约束使用剩余预算，后续评分中的性能归一化仍使用请求最初设置的 SLO。

`require_slo=True` 是正式入口默认值。开发时可在 Python 中显式使用 `require_slo=False` 做约束敏感性实验；它只关闭“指标超预算”的过滤，不会放行缺失的必需指标，也不关闭容量、健康等限制。这不是自动备用策略，默认场景也不依赖该开关。

### 3.4 二维 Pareto 与线性评分

先由 [pricing.py](../../pricing.py) 的 `resolve_request_price()` 选出本请求使用的价格档，再由 `apply_request_cost()` 计算费用。默认 `price_tier_mode="auto"` 根据预测输入 Token 的上下界或 `input_token_band` 找唯一匹配档；档位重叠、无法匹配、未知 Token 条件会排除，不会猜最低价。没有 Token 条件时使用配置的 `price_tier_index`；模态、时间等非 Token 条件仍需要调用方确定配置。`price_tier_mode="configured"` 明确指定档位也必须通过它的 Token 条件检查。

默认 `cost_mode="predicted_request"`：

```text
estimated_input_cost  = predicted_input_tokens  * input_price_per_million  / 1_000_000
estimated_output_cost = predicted_output_tokens * output_price_per_million / 1_000_000
estimated_request_cost = estimated_input_cost + estimated_output_cost
cost_raw = estimated_request_cost
```

成本单位为所选币种/次请求；例如输入 2000 Token、输出 500 Token、输入单价 2 元/百万、输出单价 8 元/百万，预计费用为 `0.004+0.004=0.008` 元。预测不等于最终实际消耗，缓存、阶梯累进、特殊模态等未建模计费项也不由本公式推断。ρ 不参与真实费用计算，否则会把实际应付的输入或输出部分人为打折。

显式兼容两种偏好模式：

| `cost_mode` | 评分中的 `cost_raw` | 含义 |
| --- | --- | --- |
| `predicted_request`（默认） | `C_input+C_output` | 本次预计费用，ρ 不生效 |
| `weighted_predicted_request` | `ρ×C_input+(1-ρ)×C_output` | 考虑长度的偏好目标，不是账单 |
| `unit_price` | `ρ×input_price+(1-ρ)×output_price` | 旧单位价格目标，长度不进入成本目标 |

无论使用哪种评分目标，候选都保留预计费用分项。候选必须具有路由所需币种的价格；当前入口不自动换汇，不混合人民币与美元。价格档解析结果、条件、来源与失败原因保存在 `endpoint_checks[].price_resolution`。

采用可用候选中的正成本中位数作为参考成本：

```text
cost_reference = median(可用候选的正成本)
cost_normalized = cost_raw / cost_reference
```

这不是 min-max 归一化，数值可以大于 `1`。若全部成本为零，则以 `1` 作为计算参考值。无可用候选时 `cost_reference` 为 `None`，图上的展示归一化不代表存在可选成本基线。

性能越低越好，计算为：

```text
非流式 performance = estimated_e2e / slo_e2e
流式   performance = η * estimated_ttft / slo_ttft
                     + (1 - η) * estimated_tpot / slo_tpot
```

`η` 控制流式首 Token 与后续生成速度的偏好。合并后的性能和成本构成二维 Pareto 图；成本与性能均不优且至少一项更差的点为被支配点，不进入首选评分选择，但满足硬约束时仍可参与稳定备用选择。

Pareto 前沿上的候选使用：

```text
score = λ * cost_normalized + (1 - λ) * performance
```

`λ = 0` 只重视性能，`λ = 1` 只重视当前选择的成本目标，中间值体现取舍。分数越低越好。正式入口确定性选择最小分数；平分时按成功率、有效样本数、Endpoint 标识稳定破平。

图中的线性等分线用于解释取舍，不是任选一条固定直线后按绝对距离选点。最优点对应向低成本、低性能值方向平移后首先触达前沿的等分线。程序没有隐含的随机抽样；请求路由入口禁止覆盖 `temperature`、输出模式与 SLO，修改 SLO 应明确修改请求本身。

### 3.5 默认三个推荐 Endpoint：首选加稳定备用

二维 Pareto 只用于成本/性能首选，不假装每次前沿都恰好有三个 Endpoint。`top_k=3` 包括一个首选及至多两个备用，候选不足就返回更短列表，且 Endpoint 不重复。

默认 `backup_pool="all_feasible"`：备用从满足容量、健康、冷却、指标与 SLO 等同样硬约束的剩余候选中选择。某 Endpoint 虽被成本/性能支配，但可靠性更好，仍可作为备用；不会把不可用点补入列表。设置 `backup_pool="pareto"` 才把备用也限制在前沿。

`estimate_stability()` 合并同一 `(model, endpoint)` 的两种流式模式最近 60 分钟尝试，按相同半衰期计算加权成功率；有限先验强度可补充证据，再以有效样本数计算 Wilson 式保守下界代理。它不混入其他模型，不使用未来完成记录。排序顺序为：

```text
稳定性下界代理降序 → 加权成功率降序 → 有效证据样本数降序
                  → routing_score升序 → endpoint_id升序
```

例如一个样本的 100% 成功率不能被当作比 99/100 的成功率更可靠。该下界只是谨慎排序代理，指数加权和先验下并非经过统计校准的真实置信保证。429/5xx 加权比例保留在候选记录中用于解释，但当前不会另加一条错误率阈值参与备用排序。

`score` 保持只有前沿点才有的旧含义；`routing_score` 给所有可用点计算同一个线性值，供备用平分使用；`route_rank`、`route_role` 表示本次计划位置。决策输出 `ordered_endpoints`、`backup_endpoints`，首选仍是 `selected_endpoint`。

### 3.6 没有可选 Endpoint 时

返回 `status="no_candidate"`、`selected_endpoint=None` 和各 Endpoint 的排除记录，`no_candidate_action="handoff_to_caller"` 将后续决定交给调用方。它并不意味着所有失败都应送入等待队列；缺少价格、指标或模型配置，通常不能靠等待自动解决。

`no_candidate_categories` 区分 `no_offering`、`temporary_capacity`、`temporary_cooldown`、`slo_infeasible`、`missing_evidence_or_configuration`、`unavailable_endpoint`、`insufficient_quality_or_evidence` 等情况。上层据此决定等待、补充配置、拒绝或其他处理。当前阶段不自动突破 SLO、不绕过容量，不假装已经实现延迟调度或重新执行请求。

优先级在场景中通过 `high → strict`、`normal → standard`、`low → relaxed` 映射到 SLO 层级，具体预算是显式可改的场景输入；正式选择器读取的是请求中已经确定的 SLO，不另外按优先级加分。`workload`、`urgency` 当前用于保持输入合理及记录解释，不直接额外加到线性分数里；重型请求决策树、延迟队列等模块尚未接入。

请求的 `input_provenance` 保存预测来源、优先级来源、SLO 的来源/画像键/版本等信息；决策中的 `active_features` 和 `metadata_only` 明确哪些信息参与选择，哪些仅用于记录。预测长度本轮用于费用、价格档与 TPM 投影；性能仍来自近期窗口和先验，并未建立按当前请求长度训练的时延预测器。

### 3.7 失败后切换：重新检查，而非盲目执行旧列表

[failover.py](../../failover.py) 封装 `FailoverSession`、`RetryPolicy`、`AttemptPlan`、`AttemptResult`。`RoutingEngine.start_failover()` 初始化会话并记录初始推荐列表；它不发送请求。

`next_attempt()` 每次以最新时间重新调用正式路由，重新检查容量、启停、冷却、健康、价格与已消耗 SLO 预算，排除之前尝试过的 Endpoint。首选仍可用时先用首选；失败后以 `selection_phase="backup"` 在初始推荐列表剩余 Endpoint 中按稳定性选，不重新变回最低评分策略，也不无限扩大备用集合。

调用方发送请求后，使用 `complete_attempt(result)` 报告真实或明确标记的 Mock 结果：

- 写入完成时刻的 `Observation`，刷新窗口与质量状态；成功则会话结束。
- HTTP 429 按 `Retry-After` 或默认 30 秒设置组合冷却，5xx 默认 10 秒；已有更长冷却不缩短。
- `retry_safe` 默认 `False`：调用方必须显式确认请求可安全重试。状态码默认可重试集合为 408/429/500/502/503/504，无 HTTP 状态的 timeout/connection_error 也可重试。
- 若已经向客户端输出任意响应内容（`response_started=True`），或错误不可重试，则停止；不在已开始的流中偷偷拼接另一个 Endpoint 的输出。
- 默认 `max_attempts=3` 包含第一次，不是额外三次。没有剩余可用候选、SLO 耗尽或预算不足也停止。

可设置 `max_estimated_total_cost` 限制累计预计尝试费用；当前按每次完整预测费用累加，并非实际失败计费。稳定性最高的可用备用超过剩余成本预算时保守停止，不再搜索便宜但不稳定的备用。

实时接入应提供 `clock` 或在每次调用显式传入 `cutoff`；历史演示用逻辑时间。规划器不负责真实 HTTP、RPM/TPM 滑动计数器或原子容量占用/释放。已有并发增减接口不等同于原子“检查并预占三种容量”。独立的轻量容量压力判断已提供 `state.assess_busy()`，详见 [繁忙判断说明](../../docs/繁忙判断说明.md)；尚未自动接入本选择器，也未实现延迟队列。

## 4. 场景怎么修改，验证什么

[selection_scenarios.json](../configs/selection_scenarios.json) 集中配置场景。请求默认值、策略参数默认值、状态模板与场景覆盖分开维护，避免把实验条件写死在路由代码里。字典递归合并，列表整体替换；具体场景数量以本次配置和输出摘要为准。

主要可修改部分为：

| 配置部分 | 作用 |
| --- | --- |
| `request_defaults` / `cases[].request` | 模型、模式、预测 Token、优先级、SLO、请求服务时间的 Mock 估计 |
| `parameter_defaults` / `cases[].parameters` | λ、η、ρ、费用模式、Top K、备用池、半衰期、样本阈值、窗口候选与先验强度 |
| `snapshot_templates` | 可复用的 Endpoint 价格、容量、负载、健康、先验和近期样本快照 |
| `cases[].endpoint_overrides` | 仅修改某个 Endpoint，如禁用、冷却、超限或指标退化 |
| `cases[].routing_delay_ms` | 到达后再做决策的等待时间，用于验证预算消耗 |

预测 Token 或 SLO 修改后，加载器重新计算场景的轻重标记和紧急程度，避免保留与新字段冲突的旧标签。分类重型信息仍为未知，不伪造已实现的分类模型。

场景检查 λ/η 改变偏好、偏好模式下 ρ 生效而默认预计账单不受 ρ 影响、预测输入/输出比例影响费用、被支配点不能作首选但可以作稳定备用、样本证据量、三个模型两种模式隔离、禁用/健康/冷却、预测容量、恰好达限、逐指标 SLO、无候选、先验、半衰期及等待预算。失败切换演示另验证首选故障反馈、组合冷却及稳定备用选择。

验证使用规则不变量，例如：首选满足所有有效限制、位于 Pareto 前沿、得分不高于其他可选前沿点；备用也满足硬约束且按稳定性排序；相同输入与固定状态重复选择一致；单次选择不修改状态。失败反馈会有意更新窗口和冷却，不能将选择器的只读约束错误套用到反馈会话。小型测试用例用于回归检查实现错误，不作为证明真实业务效果的预设“标准答案”。

## 5. 输出与路由开销

每次运行的总输出包括：

| 文件 | 内容 |
| --- | --- |
| `summary.md` / `summary.json` | 场景结果、验证结论与开销摘要 |
| `decisions.csv` | 所有场景的选择及主要指标，便于汇总查看 |
| `manifest.json` | 配置、代码、请求输入等来源及哈希，便于复现和追溯 |
| `resolved_config.json` | 本次运行的完整场景配置副本，避免原配置修改后无法还原条件 |
| `source_dataset_manifest.json` | 使用已有数据集时，保留其来源和字段来源限制 |

每个场景单独保存：

| 文件 | 内容 |
| --- | --- |
| `decision.json` | 请求、参数、价格档解析、费用分项、候选、排除原因、SLO/容量、首选/备用计划和计时 |
| `candidates.csv` | 每个候选的成本、性能、稳定性、样本、可用性、Pareto、评分与推荐位置 |
| `pareto.svg` | 可直接在浏览器打开的成本—性能图，区分首选、备用、前沿与排除点 |
| `snapshot.json` | 本场景使用的状态快照及来源，不是 Endpoint 执行日志 |
| `benchmark.csv` | 相同请求与固定状态下多次独立路由的开销记录 |

计时单位为微秒，主要阶段包括 `index_lookup`、`admission_checks`、`metric_estimation`、`pareto_and_score` 和 `route_total`。总计时包含请求校验与策略准备，但不包含绘图、文件读写或 Endpoint 请求执行。

`--benchmark-repeats` 控制固定状态下独立重复的次数；重复运行不是连续请求回放，也不逐次累积 RPM、TPM 或并发。各阶段计时及分位数用来评估当前实现的决策开销，不能据此推断生产网关在并发、网络或数据库条件下的总体吞吐量。

## 6. 从数据到失败切换：文件、函数和输出

项目分为数据准备、内存状态、正式策略、实验验证四部分；实验调用正式策略，不另写一套实验专用路由代码。

### 6.1 离线准备：完成日志变成可用于路由的初始输入

| 数据到达阶段 | 文件与函数 | 处理与输出 |
| --- | --- | --- |
| 原始完成日志 | `data/pipeline.py::prepare_dataset()` → `data/normalize.py::normalize_request()` | 清洗 ID、时间、Token、结果及尝试，去重和拒绝非法行；输出规范化 `CanonicalRequest`，保存 `canonical_requests.jsonl` |
| 训练区间 | `data/profiles.py::fit_profiles()` | 仅用截止前已完成训练样本，建立模型/模式统计画像；输出 `profiles.json`，不偷看测试请求的结果 |
| 重建请求到达信息 | `data/profiles.py::build_incoming_request()` | 从事实记录提取到达时元数据；已有真实预测保留，缺少预测用明确 Mock；输出 `IncomingRequest`，保存 `incoming_requests.jsonl` |
| 补充实验策略标签 | `data/profiles.py::enrich_request()`、`urgency_at()` | 加入三级优先级/SLO、预测 Token 轻重、预算与紧急程度；校准/测试区间输出 `requests.jsonl` |
| 分离结果与验证 | `data/schema.py::CanonicalAttempt.observation()`、`data/validation.py::validate_dataset()` | 执行结果保存为完成时刻可见的 `observations.jsonl`；验证字段关系、来源及文件哈希，输出审计/验证报告 |

原日志没有 prompt/messages，重建的是到达元数据，不声称恢复原始正文。SLO/优先级是实验规则，不是日志里真实业务标签。线上网关已经有真实请求与真实预测时，可直接构建 `RoutingRequest`，不必每次经过离线准备。

### 6.2 启动与更新：配置、观测进入内存

| 阶段 | 文件与函数 | 作用 |
| --- | --- | --- |
| 构建状态 | `memory_state.py::InMemoryRoutingState()` | 封装模型策略 `policies`、组合配置 `catalog`、观测窗口 `windows`、压力判断 `busy` |
| 注册策略与 Endpoint | `ModelPolicyRegistry.upsert()`、`InMemoryRoutingState.register_endpoint()` | 建立 `model→endpoints` 和组合状态，容量归属 `(model, endpoint)` |
| 更新组合用量/可用性 | `ModelEndpointRegistry.update_model_endpoint_load()`、`set_enabled()`、`set_cooldown()` | 写入调用方的 RPM/TPM/并发、启停和冷却；可显式调用 `state.assess_busy()` 评估容量压力，但本选择器尚未消费该信号 |
| 写入完成记录 | `InMemoryRoutingState.record_observation()` → `ModelEndpointStateIndex.record()` | 增量写入模式隔离的有界队列，刷新最近健康比例；可由完成反馈调用 |

生产不必加载完整历史日志。窗口和可选先验是用于性能估计的数据源；`models.py` 定义这些结构及预留的数据库接口，当前没有数据库实现。

### 6.3 每个请求：选择器内部的完整路径

| 顺序 | 文件与函数 | 输入、作用与输出 |
| --- | --- | --- |
| 1. 接收与校验 | `request_routing.py::RoutingRequest.from_prepared()` / `RoutingRequest.__post_init__()` | 接收画像请求或直接构建对象，校验非负整数预测、模式、SLO、优先级；输出 `RoutingRequest` |
| 2. 进入正式策略 | `routing_engine.py::RoutingEngine.route_request()` → `request_routing.py::route_request()` | 校验决策时间、模型配置与覆盖参数；模型 `cost_mode/top_k/backup_pool/stability_z` 可配置 |
| 3. 获取候选 | `memory_state.py::ModelEndpointRegistry.candidates()`、`runtime_state()`、`endpoint_state()` | 只遍历当前模型的 Endpoint，保留不可用项以记录排除原因 |
| 4. 硬准入 | `request_routing.py::_capacity_check()` 及 `route_request()` | 检查请求加入后的容量、启停、健康、冷却、允许/排除集合，输出逐 Endpoint 检查记录 |
| 5. 解析价格 | `pricing.py::resolve_request_price()` | 用预测输入匹配唯一 Token 价格档，验证计费单位、币种与非负单价；输出 `Price` 和解析记录 |
| 6. 估计性能 | `memory_state.py::recent()` → `routing_engine.py::compute_candidate()` | 读取当前模式已完成窗口，`select_window()` 选窗口，计算衰减加权 P95、有效样本与逐指标先验融合；输出 `Candidate` |
| 7. 计算费用 | `routing_engine.py::apply_request_cost()` | 预测长度乘选档单价，保存输入/输出/总费用，按显式模式决定 `cost_raw` |
| 8. 估计备用稳定性 | `routing_engine.py::estimate_stability()` | 同组合两种模式近 60 分钟成功标记，计算指数加权成功率、有限先验、保守下界代理及错误率 |
| 9. 检查 SLO | `request_routing.py::route_request()` | 分别检查 E2E/TTFT/TPOT，E2E/TTFT 扣除已等待预算；形成最终 `feasible` 和排除原因 |
| 10. 首选 | `routing_engine.py::finalize_selection()` → `pareto_mask()` | 成本除以可用正成本中位数，构建二维前沿，最小线性分数选首选；平分稳定破平 |
| 11. 备用 | `routing_engine.py::build_route_order()` → `backup_sort_key()` | 首选不变，剩余可用点按稳定性排序取至多 `top_k-1` 个；默认可保留被支配的稳定点 |
| 12. 返回解释 | `request_routing.py::RoutingDecision.to_dict()` | 输出首选、备用列表、价格/费用、窗口/先验/SLO、评分、推荐位置、原因及阶段计时；不发送请求、不改变用量 |

为解释图像，当前也计算具有有效价格但被健康/容量等条件排除的候选指标；它们只作排除点展示，不能进入首选或备用。没有价格的 Endpoint 只有检查记录，不伪造图上成本。

### 6.4 调用与失败反馈

网关启动时先初始化模型策略、Endpoint 配置和近期窗口，得到 `state`。只需要推荐列表时：

```python
from endpoint_routing_strategy import RoutingEngine, RoutingRequest

request = RoutingRequest.from_prepared(row)
decision = RoutingEngine(state).route_request(request)
if decision.selected_endpoint is not None:
    endpoint_id = decision.selected_endpoint  # 选择建议；本入口不执行发送
    backups = decision.backup_endpoints
else:
    reasons = decision.to_dict()["no_candidate_categories"]
```

容量、价格与性能窗口的初始化仍由上游维护，不在上述每次调用中重新读取历史日志。没有已配置的模型策略属于配置错误，会明确抛出异常；有模型策略但无可用Offering才是结构化 `no_candidate`。

需要管理失败切换时：

```python
from endpoint_routing_strategy import AttemptResult, RetryPolicy

# 仅在调用方能保证该请求可安全重试时开启；now 为带时区时间。
session = engine.start_failover(
    request, retry_safe=True, cutoff=now,
    retry_policy=RetryPolicy(max_attempts=3),
)
attempt = session.next_attempt(cutoff=now)
if attempt is not None:
    # 调用方实际发送、统计负载、测量结果；此处仅示意一次Mock 503。
    session.complete_attempt(
        AttemptResult(success=False, http_status=503, response_started=False),
        cutoff=failure_time,
    )
    backup_attempt = session.next_attempt(cutoff=retry_time)
```

对应函数路径为 `RoutingEngine.start_failover()` → `failover.py::FailoverSession.__init__()` → `next_attempt()` → 正式 `route_request()` 重新检查 → 调用方执行 → `complete_attempt()` 写入 `Observation`、调用 `record_observation()` 和必要的 `set_cooldown()` → 再次 `next_attempt()` 或停止。整个会话可由 `FailoverSession.to_dict()` 审计。

### 6.5 实验如何验证业务代码

`experiments/selection/scenarios.py::load_cases()` 生成初始请求与状态快照；`runner.py::run_suite()` 调用正式路由，`check_decision()` 检查硬约束、首选 Pareto/评分及稳定备用规则。`data_pipeline.py::write_svg()` 画图，结果和基准开销由 runner 保存。`failover_demo.py::run_demo()` 只注入少量 503/200 Mock 结果，验证反馈与切换，不用模拟结果证明真实延迟达标。

尚未接入：繁忙时自动速度优先与延迟队列。独立轻量容量压力判断已实现；重型请求决策树、真实 HTTP 调用、原子容量预占/释放、数据库和其他策略对照仍未实现。预测费用和可控切换已经具备，但完整生产网关闭环仍需调用方执行/计数集成。
