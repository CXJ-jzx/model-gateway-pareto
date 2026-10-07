# Endpoint 路由策略：交接与维护指南

## 1. 项目定位和当前完成度

交接基线：2026-10-07，`pyproject.toml` 包版本 `0.3.0`，Python 3.11+，当前运行仅使用标准库。本文是当前实现的交接入口；设计文档和流程图中的后续设想不等于实现承诺。

GitHub 版本只包含正式代码、测试、配置和当前文档；原始数据、实验 output、legacy、归档及图形资料仅在原工作区保留。纯合成实验可直接运行，数据相关命令需另行提供本地数据；不要把本地历史产物存在当作仓库的前提。

上游根据 prompt 选择模型；本层收到固定 `model_id`、请求模式、预测输入/输出 Token 和延迟目标，选择可提供该模型的 Endpoint。**本层不换模型，不发送 HTTP，不自动占用容量。**

| 能力                       | 状态                       | 实现及已验证边界                                      |
| ------------------------ | ------------------------ | --------------------------------------------- |
| 完成日志清洗、到达元数据重建           | 已实现                      | 输入和执行结果分离，来源/哈希/时间边界可核验；不能恢复缺失的原始 prompt      |
| Token 预测、优先级、SLO、轻重和紧急程度 | 接口/实验规则已实现               | 显式预测保留，缺失预测使用训练统计 Mock；优先级/SLO 为模拟标签，非真实业务标准  |
| 内存候选索引与动态维护              | 已实现                      | 直接取得当前模型的 Endpoint 集合；支持组合配置增删、启停、冷却和用量更新     |
| 模式隔离的近期性能窗口              | 已实现                      | 有界 deque、指数时间加权 P95、逐指标先验融合；每次决策仍会扫描内存窗口并排序统计 |
| 价格档和预计请求费用               | 已实现                      | 用预测输入匹配 Token 档，长度参与费用；同币种筛选，不自动汇率转换          |
| 硬容量、健康及请求 SLO 检查         | 已实现                      | 检查加入请求后的 RPM/TPM/并发及剩余时间预算；未知容量默认放行，可配置阻止     |
| Pareto 首选与稳定备用           | 已实现                      | 确定性最小线性评分选首选；默认至多 2 个稳定备用，不强求前沿恰好有 3 个点       |
| 失败切换规划和反馈                | 可能实现了，先忽略这个失败切换，执行为 Mock | 重查约束、消耗预算、完成观测、429/5xx 冷却及重试安全；未调用真实 Endpoint |
| 繁忙/容量压力判断                | 未完整实现，独立模块               | 组合压力与 0.8/0.6 迟滞，模型汇总；尚未触发改权重、攒批或队列，当前搁置      |

## 2. 运行与复验

### 2.1 环境

从工作区根目录 `E:\Desktop\model_gateway` 执行，而不是进入包内运行某个 `.py`。`-B` 避免生成 `__pycache__`。如需隔离环境：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\endpoint_routing_strategy\requirements.txt
```

当前 requirements 仅记录标准库实现，无需额外运行包。可用 `.\.venv\Scripts\python.exe` 替换下文 `python`，不用依赖 PowerShell 激活脚本。后续需要第三方包可以安装，但需同步维护 requirements 和 `pyproject.toml`，不能只在开发者机器安装。

### 2.2 先验证正式策略，不需要原始数据

```powershell
python -B -m unittest discover -s .\endpoint_routing_strategy\tests -v
python -B -m endpoint_routing_strategy.experiments.selection --benchmark-repeats 30
python -B -m endpoint_routing_strategy.experiments.selection.failover_demo
```

`selection` 运行配置化人工场景，输出 `experiments/output/selection_时间戳/`；失败演示注入 503→200，输出 `failover_时间戳/`。没有网络调用，也不需要 Endpoint 凭据。

使用已保存的初始请求元数据作为模板：

```powershell
python -B -m endpoint_routing_strategy.experiments.selection --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_20261003 --benchmark-repeats 30
python -B -m endpoint_routing_strategy.experiments.selection.failover_demo --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_20261003
```

重要：此入口只从数据集中取得请求到达元数据和可用模型身份；**预测 Token、SLO、优先级及 Endpoint 状态仍由场景 fixture 明确构造**。不是按真实日志逐请求重建容量与业务结果，也不是沿用数据准备阶段的 SLO 做达标评估。

### 2.3 重新准备初始请求数据

当前原始文件位于 `实验数据/历史性能数据包/`。CLI 中旧 `--traffic` 默认路径尚未迁移，本轮为保持代码不变，必须显式指定：

```powershell
python -B -m endpoint_routing_strategy.data prepare --traffic .\实验数据\历史性能数据包\流量记录_历史性能.jsonl --config .\endpoint_routing_strategy\experiments\configs\data_preparation.json
```

自动创建新的 `experiments/output/data_时间戳/`，不会覆盖基线数据。也可指定不存在或为空的 `--output-dir`。独立重验现有数据：

```powershell
python -B -m endpoint_routing_strategy.data validate --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_20261003
```

旧数据 `manifest.json` 保存的是生成时的原路径，不应手改。校验器会验证产物哈希、重算画像和关系；旧源路径不存在时不会自动搜索新路径，也不会自动验证新位置源文件。核验迁移后的源内容应单独执行：

```powershell
$datasetManifest = Get-Content .\endpoint_routing_strategy\experiments\output\incoming_data_20261003\manifest.json -Encoding utf8 | ConvertFrom-Json
$trafficHash = (Get-FileHash -LiteralPath '.\实验数据\历史性能数据包\流量记录_历史性能.jsonl' -Algorithm SHA256).Hash
$trafficHash -eq $datasetManifest.source_sha256
```

应输出 `True`；若为 `False`，不要把不同数据当作同一来源。新运行的 manifest 会记录新路径。

### 2.4 独立容量压力验证（目前不继续扩展）

```powershell
python -B -m endpoint_routing_strategy.experiments.busy_check
```

报告为 `output/busy_check_时间戳/report.json`。这只检查容量压力规则，不改变正式选路行为，不验证调度效果。

## 3. 目录组织与阅读顺序

```text
endpoint_routing_strategy/
  __init__.py                公共 API 导出
  models.py                  配置、状态、观测、候选对象及 Repository Protocol
  memory_state.py            内存索引、组合状态、模式隔离窗口
  busy.py                    独立容量压力与迟滞判断
  pricing.py                 请求价格档解析与计费校验
  routing_engine.py          窗口统计、费用、稳定性、Pareto、评分及引擎入口
  request_routing.py         RoutingRequest / RequestSLO / RoutingDecision、硬准入
  failover.py                重试安全、尝试规划和完成反馈
  data_pipeline.py           实验共用加载、候选导出、SVG；含 legacy 辅助逻辑
  data/                      离线清洗、画像、审计、初始输入、独立校验
  experiments/
    configs/                 数据规则和选择场景 JSON
    selection/               正式策略规则验证、开销与失败演示
    busy_check.py            独立压力验证入口
    legacy/                  旧单位价格/窗口扫描/历史回放，仅供参考
    output/                  已准备数据、当前报告和后续运行产物
    archive/                 此前生成结果的压缩归档，不是本轮源码基线
  docs/                      状态、压力说明、历史阶段说明及实验路线图
  tests/                     数据/价格/窗口/准入/备用/反馈/入口回归测试
  requirements.txt           依赖记录
```

建议代码阅读顺序：`models.py` → `memory_state.py` → `request_routing.py` → `routing_engine.py` → `failover.py`；再看 `data/` 和实验 runner。不要将 `experiments/legacy/router_strategy.py` 或旧 `run_demo.ps1` 当作正式入口。

工作区其他文件、数据与图的位置见 [根目录导航](../README.md)。本地交接归档说明为 `archives/README.md`，归档不上传 GitHub。

## 4. 按数据流程理解实现

### 4.1 离线：完成日志 → 到达输入 → 可验证产物

| 阶段      | 文件与函数                                                                          | 处理与输出                                                            |
| ------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| 读取完成日志  | `data/pipeline.py::prepare_dataset()`、`data/normalize.py::normalize_request()` | 校验时间/ID/模式/Token，保留实际结果与 attempt；形成 `CanonicalRequest`，非法行进入拒绝记录 |
| 按到达时间切分 | `prepare_dataset()`                                                            | 训练 60%、校准 20%、测试 20%；相同到达时间不跨切分，先验只用训练截止前完成结果                    |
| 拟合冻结画像  | `data/profiles.py::fit_profiles()`                                             | 按模型/模式统计 Token 中位数、延迟 P75 和轻重阈值；样本不足回退全局或配置                      |
| 重建到达信息  | `build_incoming_request()`                                                     | 提取模型、ID、时间、模式；真实预测保留，缺失预测明确 Mock；不带当前请求的最终结果                     |
| 添加实验标签  | `enrich_request()`、`urgency_at()`                                              | 增加模拟优先级、SLO、预算、轻重与时间余量；校准/测试形成增强输入                               |
| 分离完成观测  | `data/schema.py::CanonicalAttempt.observation()`                               | 生成完成后才能写入窗口的 `Observation`；不是请求到达即公布延迟                           |
| 独立复验    | `data/validation.py::validate_dataset()`                                       | 检查数量、哈希、字段关系、防泄漏，重算画像与标签，输出验证结论                                  |

主要产物：

| 文件                                      | 交接含义                                                           |
| --------------------------------------- | -------------------------------------------------------------- |
| `incoming_requests.jsonl`               | 到达元数据：ID、模型、到达时间、模式、预测 Token 和来源等；不含最终 Endpoint/状态/实际 Token    |
| `requests.jsonl`                        | 校准/测试的增强输入：另含优先级、SLO、workload、urgency 和来源；可转为 `RoutingRequest` |
| `canonical_requests.jsonl`              | 执行后规范事实；可离线核对，不能作为当前请求选路输入                                     |
| `observations.jsonl`                    | attempt 完成观测；按完成时间发布，供窗口更新                                     |
| `profiles.json`、`resolved_config.json`  | 冻结训练画像、实际配置                                                    |
| `audit.json`、拒绝 JSONL、`validation.json` | 缺失/清洗/关系检查及验证结论                                                |
| `manifest.json`、`summary.md`            | 来源与产物哈希、切分边界、数量及限制                                             |

原始流量日志没有可关联的完整 prompt，独立 prompt/回答包也没有可直接连接的请求 ID；不应按行号强行拼接。缺失预测不是把当前实际 Token 改名为预测。当前 Mock 主要为流程占位，不证明预测准确性，普通输入中轻请求/normal 紧急程度占多数不代表清洗错误。

### 4.2 运行时：配置和观测 → 内存状态

`InMemoryRoutingState` 封装四个对象：

| 成员                                 | 维护内容                                                                           | 主要更新接口                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `policies: ModelPolicyRegistry`    | `model_id → ModelRoutingPolicy`，λ/η/ρ、默认目标及费用/备用设置                             | `policies.upsert()`                                                                                                     |
| `catalog: ModelEndpointRegistry`   | `model → endpoints`、组合 Offering/容量/当前用量/价格档/先验/启停/冷却/质量；另保留 Endpoint 整体健康与支持模型 | `register_endpoint()`、`remove_endpoint()`、`update_model_endpoint_load()`、`set_prior()`、`set_enabled()`、`set_cooldown()` |
| `windows: ModelEndpointStateIndex` | `(model_id, endpoint_id, is_stream) → deque[Observation]`                      | `record_observation()`、`bulk_record()`、`recent()`                                                                       |
| `busy: BusyDetector`               | 组合容量压力与进入/退出阈值记忆，不复制用量计数                                                       | `state.assess_busy()`                                                                                                   |

RPM、TPM、最大并发和当前用量均属于 **`(model, endpoint)`**；Endpoint 聚合用量不是组合准入依据。用量由调用方更新，代码尚不自动从事件累计 RPM/TPM。

窗口默认最多 360 分钟、每键 20,000 条；写入时按最新观测时间裁剪，乱序记录会排序。查询再按决策时间过滤，不包含未来完成记录；长期无新写入时旧记录可能仍驻留，但不参与超窗查询。统计是内存扫描，不是每请求重新读磁盘日志，也不是已经缓存好所有 P95。

`Prior` 当前挂在组合状态上，没有独立的流式/非流式先验索引；对象可含 E2E/TTFT/TPOT 指标，接入历史快照时仍需审查口径，不能将混合模式统计当作模式专属时延。

### 4.3 在线元数据：请求 → 首选和备用

```text
RoutingRequest
→ 模型候选索引
→ 启停/健康/冷却/预计容量检查
→ 价格档解析
→ 当前模式窗口、性能与费用估计
→ 请求剩余 SLO 过滤
→ Pareto 前沿与线性评分选首选
→ 稳定性排序选备用
→ RoutingDecision（建议，不执行）
```

| 顺序  | 文件 / 函数                                                                                    | 实现作用                                           |
| --- | ------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| 1   | `request_routing.py::RoutingRequest.from_prepared()` / 构造器                                 | 检查预测非负整数、时间带时区、模式和 SLO；不接收执行后字段作为到达输入          |
| 2   | `routing_engine.py::RoutingEngine.route_request()` → `request_routing.py::route_request()` | 正式入口，读取模型策略；默认逻辑时间为到达时间，线上/等待后应显式提供当前 `cutoff` |
| 3   | `memory_state.py::ModelEndpointRegistry.candidates()`                                      | 通过索引只遍历该模型的 Endpoint；包括不可用点以保留排除解释             |
| 4   | `request_routing.py::_capacity_check()`、`route_request()`                                  | 检查加入本请求后的 RPM+1、TPM+预测输入输出、并发+1，及启停/整体健康/组合冷却  |
| 5   | `pricing.py::resolve_request_price()`                                                      | 用预测输入选择唯一价格档，校验币种/计费单位/价格；没有或多档命中则记录拒绝原因       |
| 6   | `memory_state.py::recent()` → `routing_engine.py::compute_candidate()`                     | 取模式隔离的近期完成记录，选择窗口、加权 P95、逐指标融合先验与证据检查          |
| 7   | `routing_engine.py::apply_request_cost()`                                                  | 计算输入/输出/总预计费用，按费用模式设置成本目标                      |
| 8   | `routing_engine.py::estimate_stability()`                                                  | 同组合两种模式近 60 分钟成功/失败，估计稳定性排序代理                  |
| 9   | `request_routing.py::route_request()`                                                      | TTFT/TPOT/E2E 逐项硬过滤；E2E/TTFT 扣掉从到达到决策的已耗时间     |
| 10  | `routing_engine.py::finalize_selection()` → `pareto_mask()`                                | 对可用点归一化费用，构建成本—性能前沿，确定性最小评分选首选                 |
| 11  | `build_route_order()` → `backup_sort_key()`                                                | 剩余硬约束可用点按稳定性取至多 `top_k-1` 个；默认包含可用但被二维支配的可靠点   |
| 12  | `RoutingDecision.to_dict()`                                                                | 返回首选、备用、排除原因、指标/费用/样本/SLO 和阶段耗时；无候选返回结构化结果     |

若模型策略未注册，会报配置错误；已注册模型但没有可用候选，返回 `status="no_candidate"`，由调用方处理。不会自动放宽 SLO，也不能认为所有无候选情况都适合排队等待。

### 4.4 失败后的规划与反馈

`RoutingEngine.start_failover()` → `failover.py::FailoverSession`：

1. `next_attempt()` 按最新时间重查约束和预算；第一次用首选，后续在原推荐列表的未尝试 Endpoint 中按稳定性优先。
2. 调用方发送请求并更新负载；当前演示仅 Mock，不发送 HTTP。
3. `complete_attempt(AttemptResult)` 写入完成观测和质量，429 按 Retry-After/默认 30 秒冷却，5xx 默认 10 秒；成功结束。
4. 已向客户端输出内容、不可重试错误、不安全请求、预算耗尽或无可用备用均停止。

`retry_safe` 默认 False，必须由调用方显式确认；默认最多 3 次尝试包含第一次。累计费用限制是完整预测尝试费用之和，不是失败实际账单。实时集成应传 `clock` 或每次最新 `cutoff`；并发增减接口不等于原子检查并预占容量。

## 5. 策略计算的核心公式

### 5.1 近期性能

默认窗口候选 5/15/30/60/180/360 分钟，目标尝试数 100；选达到目标的最短时间窗口，全部不足取最大窗口。窗口保留全部记录，不是固定最近 100 条；失败记录计入尝试数，但只有成功且有效的指标参与分位数。

```text
w_i = 2^(-样本年龄分钟 / half_life_minutes)     # 默认半衰期 30 分钟
n_eff = (Σw_i)^2 / Σ(w_i^2)                 # 每个指标分别计算
alpha = prior_strength / (n_eff + prior_strength)
estimated_metric = (1-alpha) × weighted_P95 + alpha × prior_metric
```

默认先验强度 30。指标缺失时使用可用先验，先验缺失时使用近期值；无先验且必要证据不足可能拒绝。加权分位数为排序后累计权重达到 95% 的值。融合是启发式点估计，**不是合并分布后的真实 P95、不是递推 EWMA、不是针对当前预测长度训练的时延模型**。窗口先按尝试数确定，不会为某个缺失指标自动重新扩窗；有效样本数也不能单独证明观测足够新鲜。

正式请求入口以逐指标融合值构造性能及硬 SLO 检查；`compute_candidate()` 还保留 legacy 复合性能字段。扩展时需区分正式入口与旧 `RoutingEngine.route()`。

### 5.2 费用、性能、首选

```text
预计费用 C = (predicted_input × input_price + predicted_output × output_price) / 1,000,000
C_normalized = C / 可用候选正成本中位数
非流式 P = estimated_e2e / request_slo_e2e
流式 P = η × estimated_ttft / request_slo_ttft
       + (1-η) × estimated_tpot / request_slo_tpot
score = λ × C_normalized + (1-λ) × P
```

成本和性能越小越好；首选必须硬约束可用且在 Pareto 前沿，再选择最低 score。λ 偏向成本、η 偏向 TTFT；即使 η=0/1，另一个流式指标仍单独做硬 SLO 检查。分数相同用成功率、证据量、Endpoint ID 稳定破平。

归一化不是 0–1 min-max；成本可以大于 1。正数统一缩放不改变同目标的支配关系，但会影响线性权衡尺度。图上的线是等分线，不是按距离任意固定线选点，正式入口不随机抽样。

默认 `cost_mode="predicted_request"`，ρ 不改变预计账单。另支持显式 `unit_price` 和 `weighted_predicted_request` 偏好目标；后者含ρ，不应称为真实费用。当前只接受请求指定币种的候选，没有自动人民币/美元转换，跨币种对比需另外引入汇率来源和版本。

### 5.3 硬约束和备用

组合准入使用 `current + predicted_increment <= configured_limit`，恰好达到上限允许；0 表示不可用，None 表示未知。默认 `unknown_capacity_policy="allow"` 并记录未知，可设为 block；压力模块的 unknown 不等于准入模块的默认放行。

E2E/TTFT 的剩余预算等于请求阈值减已耗时间；TPOT 是每 Token 时间上限，不减等待时间。现有硬检查是历史估计的筛选规则，不保证每次真实执行达标，也不自动取消超时请求。

备用排序：稳定性下界代理降序 → 加权成功率降序 → 有效证据量降序 → routing_score 升序 → ID。下界使用加权有效样本及有限先验的 Wilson 式计算，未校准为真实置信保证。默认 `top_k=3` 包含首选，`backup_pool="all_feasible"`；候选不足不凑数。

### 5.4 当前 SLO 来源：必须与后续方案区分

存在三种不同口径，不应混为一谈：

| 来源                                   | 当前用途                                                                                                |
| ------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `data/profiles.py::enrich_request()` | 用训练 P75、预测 Token 和倍率生成实验 SLO；high/normal/low 对应 strict/standard/relaxed，倍率 1.5/2/3；排队比例 10%/20%/30% |
| `selection_scenarios.json`           | 人工场景显式 SLO，用来验证规则；不来自真实达标承诺，也不使用上述增强数据的阈值                                                           |
| `ModelRoutingPolicy` 默认值             | legacy 模型级默认和策略参数模板；正式请求使用 `RequestSLO` 给定阈值                                                        |

服务时间缩放、0.25 下限、三档和优先级绑定均属于未校准的旧实验规则，不是后续必须保留的业务定义。`max_wait_ms`、轻重、紧急程度未接入排队或额外评分。源 TTFT 是首响应代理，TPOT 是近似派生；Endpoint attempt 时延不等于整个网关请求耗时。

## 6. 已有验证证据与局限

原工作区保留结果索引为 `experiments/output/README.md`；以下是本地交接基线记录，结果文件不随 GitHub 仓库上传，可通过上述纯合成入口重新生成：

| 证据                                   | 基线结果                                                                 | 能证明什么                             |
| ------------------------------------ | -------------------------------------------------------------------- | --------------------------------- |
| `incoming_data_20261003/`            | 原始 24,559 行，接受/初始输入 24,524，拒绝 35，训练 14,714，增强 9,810，观测 25,319；产物验证通过 | 数据处理与字段关系、防泄漏规则可重现；不证明真实预测准确      |
| `selection_20261006_103330_525051/`  | 33 场景通过，31 selected、2 no_candidate                                   | 三个模型身份、两种模式及各项路由规则；Endpoint 为人工快照 |
| `failover_20261006_103334_353468/`   | 首选 Mock 503 → 稳定备用 Mock 200                                          | 反馈、冷却、剩余预算和安全切换逻辑                 |
| `busy_check_20261006_121425_348657/` | 10/10 通过                                                             | 组合压力、迟滞及模型汇总，不证明时延/吞吐改善           |
| 本轮回归测试                               | 185 项，183 通过、2 跳过                                                    | 已执行用例的回归；两个原始配置价格测试仍找旧根数据路径而跳过    |

场景验证检查规则不变量：可用性、最小前沿评分、备用排序、重复一致、选择不改状态。不是用真实反事实结果证明选择“最优”。固定状态重复路由计时包括索引、准入、指标、Pareto/评分，不含 I/O/绘图/真实执行；它不是请求连续回放，也不累加用量。

选择实验的模型映射检查两模式是否有输入，但不是严格的分桶统计样本充分性筛选。后续 SLO 实验要另做每个模型/模式/长度桶的证据审查。

发布版测试不强制依赖旧实验：没有 legacy 时跳过两个旧入口检查，正式清洗/路由仍执行；没有原始数据时两个原始价格案例也跳过。纯代码 checkout 预计 185 项中 181 项通过、4 项跳过，最终以实际运行输出为准。

## 7. 尚未完成的工作

| 工作                 | 已有支点                             | 还缺少什么                                               |
| ------------------ | -------------------------------- | --------------------------------------------------- |
| 可信 SLO 定义与验证       | RequestSLO、清洗器、训练切分              | 条件分位数、样本/覆盖率、口径一致、校准及冻结测试、明确达标分母；下一阶段独立开展           |
| 真实输入/输出预测          | 到达预测字段和来源契约                      | 与上游真实预测器对接、误差及长度分桶误分配验证                             |
| 当前长度相关 Endpoint 时延 | 模式隔离窗口及先验                        | 输入/输出/缓存等条件统计或预测；目前窗口混合长短请求                         |
| 自动当前容量维护           | 组合用量更新和并发增减                      | 请求事件到分钟滑动 RPM/TPM、在途 Token、原子检查/预占/释放及并发一致性         |
| 真实请求闭环             | FailoverSession/AttemptResult    | HTTP/SDK、凭据、超时/取消、真实 TTFT 采集、计费与反馈测量                |
| 繁忙消费和延迟调度          | 独立 BusyDetector                  | 模式切换、短攒批窗口、同模型新旧请求调度、饥饿保护；当前搁置，不扩展                  |
| 完整重型决策树            | Token 阈值 workload                | 输入重型/输出重型进一步业务判别，classification_heavy 特征和模型         |
| 数据库/跨进程状态          | ModelEndpointRepository Protocol | 适配器、持久化、共享缓存/一致性；Protocol 仅覆盖 Offering，不是所有状态的数据库接口 |
| 实际收益/其他常见策略对照      | 独立场景和报告框架                        | 公平的同输入对比、真实或受控结果、达标率/费用/重试率；本阶段未接入其他策略              |

不要从成功样本的延迟 P95 推断总请求成功且达标率；失败、取消、缺失指标需单独定义分母和统计覆盖。

## 8. 后续如何修改

### 8.1 常见修改位置

| 修改目的          | 优先修改位置                                                                              | 验证要求                                   |
| ------------- | ----------------------------------------------------------------------------------- | -------------------------------------- |
| 清洗规则/原始字段     | `data/normalize.py`、`data/schema.py`                                                | 数据契约测试、数量守恒、时间与缺失口径                    |
| 到达预测/旧标签规则    | `data/profiles.py`、`experiments/configs/data_preparation.json`                      | 不使用当前结果或测试未来数据；重算和来源验证                 |
| 状态字段/候选维护     | `models.py`、`memory_state.py`                                                       | model/endpoint/mode 隔离、删除联动、线程一致性与未知语义 |
| 价格规则          | `pricing.py`                                                                        | 档位边界、重叠/未命中、币种、单位；不要改费用口径来迁就案例         |
| 窗口、衰减、成本/性能目标 | `routing_engine.py`                                                                 | 正式请求与 legacy 区别、逐指标证据、公式及备用回归          |
| 硬 SLO/容量和请求字段 | `request_routing.py`                                                                | 剩余预算、恰好达限、缺失值、无候选解释                    |
| 失败重试/反馈       | `failover.py`                                                                       | 不安全/已输出请求不切换，最新时间重查、尝试数、冷却             |
| 新场景和开销测量      | `experiments/configs/selection_scenarios.json`、`selection/scenarios.py`、`runner.py` | 显式 fixture/来源，结果新目录，验证规则而非伪造真实收益       |

修改流程：先写清楚目标/字段/公式 → 添加对应测试或配置案例 → 小范围实现 → 全量测试 → 新输出目录验证 → 更新 README 和来源。旧 manifest 和已归档产物不可手改为“通过”。

### 8.2 下一阶段独立 SLO 小实验：仅规划，尚未创建

建议新建工作区根目录 `slo_experiments/`，与 `endpoint_routing_strategy/` 并列，按“README、configs、少量分析/验证文件、tests、output”组织，不复制整个项目，不写入当前包的配置或产物。

边界与顺序：

1. 只读 `实验数据/`，必要时复用已核验的规范化函数；冻结输入、配置、代码版本与口径。
2. 选择样本足够且有长短请求的模型，先做模型/模式/Token 长度条件性能画像，检查 Endpoint、输入长度、缓存等混杂因素。
3. 非流式先检查输出长度分桶 E2E；流式分别验证 TTFT 与输入长度、TPOT 与长度的关系，不机械地全部按输出长度分桶。
4. 训练拟合候选分位点、校准选择阈值、冻结测试评估；样本不足合桶/回退，不用测试集调阈值，也不强制三档。
5. 历史实际输出可用于诊断；在线给请求分配目标必须使用到达预测或输出预算，验证时计入预测误分桶影响。
6. 区分成功样本条件分位数、总成功率、指标覆盖率与流式联合达标率；两指标分别 P95 不等于联合 95%。代理指标不直接包装成真实客户承诺。
7. 若同一请求不同 Endpoint 的性能需要长度条件化，先检查当前混合窗口与新 SLO 的比较是否匹配。
8. 实验通过后再设计单独适配层生成 `RequestSLO`，显式选择旧/新策略来源和版本；未获确认不替换当前 `enrich_request()`。

模型 S 有两模式和长短样本但观测 Endpoint 较单一，更适合画像；BP 有多个 Endpoint，但非流式长输出样本较少，需要另评估/合桶。这是探索线索，不是最终阈值或最终模型选择结论。

`slo_experiments/` 是独立实验目录，不等于 Git 分支。交接归档时尚未初始化 Git；发布阶段建立 Git 管理，并使用忽略规则排除数据、产物与本地历史资料。后续独立 SLO 目录准备发布时须显式调整根目录白名单。

## 9. 交接资料导航

- [数据字段、旧 SLO 规则和防泄漏](data/README.md)
- [正式选择公式、场景、输出和失败切换](experiments/selection/README.md)
- [内存状态详细字段](docs/状态维护总结.md)
- [独立容量压力说明](docs/繁忙判断说明.md)
- [实验路线图：包含尚未实现内容](docs/分步骤实验方案.md)
- [实验入口](experiments/README.md)
- 仅本地：`experiments/output/README.md`、`experiments/legacy/README.md`、`experiments/archive/README.md` 和工作区 `archives/README.md`；对应数据/历史文件不上传。

若历史阶段文档或图示与本文冲突，以当前代码、测试和本交接基线的范围为准，再更新资料；不要靠更改文档将计划写成已实现。
