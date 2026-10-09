# BP SLO、Pareto与Endpoint路由：交接及系统接入指南

更新日期：2026-10-08。本文以**20%非流式SLO邻域＋近期加权均值＋前沿优先补位/低证据兜底**为当前基线。历史规则单列，不与默认实现混用。

快速跳转：[运行](#3-运行与复验) · [数据契约](#4-数据契约和时序) · [SLO](#5-当前slo设计) · [性能与索引](#6-成本性能与内存索引) · [候选策略](#7-候选策略与返回契约) · [实验结果](#9-当前实验结果和限制) · [系统融合](#10-融合完整系统的分步方案)。

## 1. 定位与完成度

上游已确定model_id，提供模式与预测输入/输出tokens。本层分配请求SLO、选择提供该模型的Endpoint。目前只研究**模型BP**和配置中的**端点C/D**，是可审计离线实验，不是已接通真实服务的网关。

本项目不导入旧 `../endpoint_routing_strategy/`，也未修改其路由。两个项目都有RoutingEngine，但构造、状态、输入和返回契约不同，不能互换。

| 能力 | 状态 | 验证边界 |
|---|---|---|
| BP提取、完成事实规范化、画像 | 已实现 | 保留失败/缺失/长尾与来源；完成日志不是初始请求 |
| 70/30时间切分、SLO训练和覆盖验证 | 已实现 | 分模式，无校准；训练标签须在测试起点前完成 |
| 长度相关SLO、稀疏/超长/冷启动兜底 | 已实现 | 有统计与风险标记，尚非正式业务承诺 |
| 模型索引、价格档、费用、显式汇率 | 已实现 | 预测长度参与成本，不猜缓存折扣 |
| 完成缓存、长度索引、均值/P50 | 已实现 | 历史因果回放，不是训练的Endpoint时延模型 |
| SLO门槛、Pareto、补位、低证据候选 | 已实现 | 至多3位，不按稳定性补位，风险不能丢弃 |
| 请求图、开销、完整重算 | 已实现 | 证明规则执行，不证明真实收益 |
| 真实token预测、HTTP及重试执行 | 未接入 | 预测为约定mock，候选不执行调用 |
| RPM/TPM/并发、健康/冷却准入 | BP实验未接入 | 配置有容量字段不等于已执行容量检查 |
| 原子预占、忙时调度、数据库 | 未实现 | 需融合阶段补齐；繁忙机制目前搁置 |
| 多模型、生产SLO及其他策略收益 | 未验证 | BP限定需泛化，新时间段与闭环验证缺失 |

交接顺序：第3节运行，第4–7节当前规则，第8节接口，第9节结果，第10节融合。旧说明已备份至 `../archives/readme_handoff_20261008/bp_slo_lab_README.md`，不必由历史README猜最新规则。

## 2. 目录与代码职责

~~~text
bp_slo_lab/
├─ README.md / pyproject.toml / requirements.txt
├─ configs/
│  ├─ slo_neighborhood.json       当前SLO：20%输出邻域、流式参考及兜底
│  ├─ routing_available.json      当前路由：均值、1.2倍率、补位和低证据
│  ├─ slo_tolerance.json          独立10%覆盖评估容差
│  └─ analysis.json 等            画像和历史固定分桶/P95配置
├─ bp_slo/
│  ├─ __main__.py                 CLI、产物组织、验证分派
│  ├─ dataset.py                  提取、事实/指标规范化、哈希读写
│  ├─ statistics.py               分位数、方差、相关性、画像
│  ├─ slo.py                      70/30、mock请求、覆盖评估；旧分桶兼容
│  ├─ neighborhood_slo.py         当前训练索引、SLO查询、兜底
│  ├─ neighborhood_experiment.py  当前SLO实验、图、边界、重算
│  ├─ tolerance.py                独立评估容差，不是新版路由倍率入口
│  ├─ performance.py              完成缓存、长度桶、统计与证据
│  ├─ routing.py                  模型配置、价格、RoutingEngine分派
│  ├─ availability.py             当前完整候选策略
│  ├─ pareto.py                   纯非支配判断、score、可选补位
│  ├─ routing_experiment.py        抽样、回放、图、计时、重算
│  └─ report.py / plots.py / validation.py / slo_experiment.py
├─ tests/                        合成规则、时序、篡改、回放、兼容测试
└─ runs/                         每轮独立数据/规则/结果，不覆盖，不默认提交
~~~

| 数据阶段 | 关键函数 | 输入→输出 |
|---|---|---|
| 提取 | dataset.extract()→normalize_request() | 三份原始JSONL→BP完成事实、配置、历史引用 |
| 训练SLO | neighborhood_experiment.prepare()→run_experiment()→fit_profiles() | 事实/配置→索引规则、模拟测试输入、覆盖 |
| 查询SLO | NeighborhoodSLO.assign() | 初始请求特征→三档原始目标及证据 |
| 配置索引 | ModelEndpointCatalog | offering→model索引和(model,endpoint)记录 |
| 更新性能 | PerformanceIndex.advance() | 到达时间→过去完成记录入缓存、过期清理 |
| 估计性能 | PerformanceIndex.estimate() | 请求/Endpoint/指标→均值/P50、样本证据 |
| 预测成本 | ModelEndpointCatalog.quote() | 请求/Endpoint/货币/汇率→费用及唯一价格档 |
| 路由 | RoutingEngine.route()→availability.route_available() | 请求→decision、timing |
| 排名 | pareto.rank_frontier() | 有效成本/损失点→前沿、score、至多3位 |
| 实验 | select_requests()→run_replay()→prepare_replay() | 冻结事实/规则/配置→结果目录 |
| 复验 | neighborhood_experiment.validate()/validate_replay() | 结果包→哈希、重算、时序、计时检查 |

routing.py在存在selection_policy时调用availability.py，否则保留旧P95严格路径。rank_frontier()单独调用默认仍是前沿-only；当前策略**显式传入fill_dominated=True**。

analysis.json中的60/20/20及前沿-only契约只是早期画像诊断，不是当前SLO训练或路由。当前70/30由slo_neighborhood.json决定；slo_experiment.json、routing_experiment.json、routing_neighborhood.json仅作历史复验。

## 3. 运行与复验

### 3.1 环境和无私有数据测试

Python 3.11+。运行依赖matplotlib>=3.7,<4，记录于本目录requirements与pyproject；根目录pyproject属于另一包。

~~~powershell
cd E:\Desktop\model_gateway\bp_slo_lab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -q
~~~

以下用已安装依赖的python，也可替换成.\.venv\Scripts\python.exe，无需激活脚本。最近139项测试通过，交接以重跑为准；-B不生成缓存。

### 3.2 已有本地结果包

~~~powershell
python -B -m bp_slo validate .\runs\bp_neighborhood_slo_relative20_20261008
python -B -m bp_slo validate .\runs\bp_routing_relative20_20261008

# 自动新建时间戳目录，使用当前20%基线及routing_available.json。
python -B -m bp_slo routing-replay
~~~

默认baseline为runs/bp_neighborhood_slo_relative20_20261008，offerings为runs/bp_baseline_20261007/dataset/endpoint_offerings.jsonl。只有Git代码、没有上述runs时不能直接依赖默认命令，需先重建或指定--baseline/--offerings。

### 3.3 从原始数据完整重建

源目录需包含：流量记录_历史性能.jsonl、端点配置.jsonl、历史性能快照.jsonl。快照提取用于审计；当前SLO和性能不拿重叠快照扩充独立样本。

~~~powershell
# 输出目录必须不存在，重跑请换名称，不覆盖旧实验。
python -B -m bp_slo prepare --source-dir "..\实验数据\历史性能数据包" --output-dir .\runs\handoff_data_01

# 内部直接做当前70/30；不必先运行旧slo-experiment。
python -B -m bp_slo neighborhood-slo --input .\runs\handoff_data_01\dataset\requests.jsonl --config .\configs\slo_neighborhood.json --output-dir .\runs\handoff_slo_01

python -B -m bp_slo routing-replay --baseline .\runs\handoff_slo_01 --offerings .\runs\handoff_data_01\dataset\endpoint_offerings.jsonl --config .\configs\routing_available.json --output-dir .\runs\handoff_routing_01

python -B -m bp_slo validate .\runs\handoff_data_01
python -B -m bp_slo validate .\runs\handoff_slo_01
python -B -m bp_slo validate .\runs\handoff_routing_01
~~~

修改源配置不会改变冻结的config.json/规则。改SLO后要生成新包并显式指向新baseline；仅改配置后跑默认routing，仍使用指定的旧冻结基线。validate只读，失败返回非零。

从工作区根目录导入BP模块，需要单独安装包，如python -m pip install -e .\bp_slo_lab；安装只解决模块路径，不等于系统融合。

## 4. 数据契约和时序

### 4.1 初始请求：到达时可用

当前route()消费字典：

~~~python
request = {
    "request_id": "example_001",
    "model_id": "模型BP",
    "stream_type": "nonstream",  # 或stream
    "arrived_at": "2026-09-21T09:49:40+00:00",  # 带时区ISO时间
    "predicted_input_tokens": 2878,
    "predicted_output_tokens": 654,
}
~~~

tokens为非负整数。当前grade取路由实例配置、默认relaxed，**不会读取请求里的slo_grade**；需要请求级档位时须新增明确适配，不能多加字段就当功能生效。

测试输入/输出预测分别由实际tokens加固定种子20261007的±50整数扰动产生，截断到0，缺失保持null。这是约定mock，不是真实预测器；上线需由上游提供真实预测，并记录版本/来源。

### 4.2 完成事实：结束后才可用

字段包括finished_at、success、final_endpoint_id、actual_input/output_tokens、attempt_count、指标和quality_flags。真实延迟、成功结果和最终Endpoint不得进入当前请求特征。

| 指标 | 单位 | 当前口径 |
|---|---|---|
| request_e2e_ms | ms | 网关收到请求至完成 |
| ttft_proxy_ms | ms | 最终尝试首响应代理，不是已认证的用户首token时间 |
| tpot_proxy_ms | ms/token | 最终尝试(耗时−TTFT)/(输出tokens−1)，输出须>1 |

缺失不补0，失败不混入成功分位数，本地补估tokens不进当前参考。Endpoint性能另要求单次尝试成功，避免把重试总时间归给最终Endpoint。方差用于离散度描述，不直接进score。

### 4.3 两种历史边界

- SLO：流式/非流式按到达时间各70/30，同一边界到达不拆开，训练标签必须在该模式测试起点前完成。规则随后冻结。
- 性能：完整日志按完成事件推进，较早测试请求已结束可用，但当前/未来结果不可用。历史来自原路由，不假设新策略反馈相同。

mock刻意用到了实际长度；只能说在约定mock输入前提下性能估计不读当前真实延迟，不能因此声称具有真实上游预测质量。

## 5. 当前SLO设计

fit_profiles()训练，NeighborhoodSLO.assign()查询。参考合并BP所有Endpoint；请求目标不等于某Endpoint当前性能。查询对象构建一次缓存长度数组，不逐请求重训。

### 5.1 非流式：输出长度前向邻域

按(actual_output_tokens,完成时间倒序,request_id)排序：

~~~text
Δ = max(200, ceil(0.2×预测输出L))
区间 = [L, L+Δ]
bisect_left(L) / bisect_right(L+Δ)
取区间开头最多20条，优先较近长度；等长优先较新记录
~~~

- ≥5条：strict=P50、standard=P75、relaxed=P90。
- 1–4条：局部P90基准×1/1.3/1.5，标记稀疏；不是分位数置信保证。
- 历史范围内没有前向样本：全训练样本线性参考。
- 超过历史最大长度：最长20条尾部线性外推并标风险。
- 缺预测：模型全局P90；指标完全无历史：工程冷启动。

线性参考b=max(0,OLS斜率)，a=max(0,P90(E2E−b×tokens))，基准a+b×L；斜率为0时用观察到的每token耗时P75。尾部外推不低于尾部E2E P90，超过训练最大长度2倍标证据域，不截断长任务时限。

tokens排序不能直接得延迟分位数，仍计算选中值。20%/200/20条/5条可调但不是可信度证明；单侧取样可能保守。扩大范围不一定改变已取满20条的样本。

标签0–512、513–2048、2049–8192、8193–32768、32769+仅作分组，不阻止跨桶。非流式25条>8192已进入训练，最大21759；本轮测试实际最大3334，**>8192处理有逻辑，但无该测试集覆盖证据**。

### 5.2 流式TTFT

预测输入I查固定[I,I+200]，最多20条；至少5条时：

~~~text
α = n/(n+20)
base = max(全局P90, α×局部P90+(1−α)×全局P90)
三档 = base×[1,1.3,1.5]
~~~

不足保留全局P90；超出输入历史域标记，但不机械线性外推TTFT。全局底线避免小样本过严，也可能让快请求预算过宽。

### 5.3 流式TPOT

全部有效流式训练记录的P90×1/1.3/1.5。当前498条、P90约33.089ms/token；三档原始33.089/43.016/49.634ms/token。**TPOT strict不是P50**。

TPOT目标不随请求tokens分段；Endpoint实际性能仍按输入长度独立估计。498是模型SLO训练参考数，不是路由表22/16等Endpoint近期样本量。不可拿全局值冒充每个Endpoint性能。

### 5.4 冷启动和两个容差路径

指标完全无历史时原始工程默认如下，标记business_approval_required、guarantee=false：

| 指标 | strict | standard | relaxed |
|---|---:|---:|---:|
| 非流式E2E | 10000ms | 13000ms | 15000ms |
| 流式TTFT | 5000ms | 6500ms | 7500ms |
| 流式TPOT | 100ms/token | 130ms/token | 150ms/token |

某Endpoint缺近期记录不使用这些数字伪装性能，而是unknown候选。

| 路径 | 公式 | 用途 |
|---|---|---|
| 原始SLO | T | 规则给请求分配的目标，不改写 |
| 覆盖评估 | T+max(T×0.1,0)=1.1T | 在副本上比较历史实际延迟 |
| 最新路由 | T×slo_limit_multiplier=1.2T | 均值/P50门槛及损失归一化 |

T=1，评估1.1、路由1.2，**不是1.32**。流式/稀疏档位1.5属于SLO设计，最宽路由仍可能是base×1.5×1.2=1.8base；不能因此称完全没有多层放宽。

三档名称不等于未来覆盖概率。形成业务承诺前需明确时限、覆盖目标、失败率和适用范围。

## 6. 成本、性能与内存索引

### 6.1 配置和成本

ModelEndpointCatalog维护model_to_endpoints[model]→set及records[(model,endpoint)]→offering，支持upsert/remove；定位当前模型后只遍历它的Endpoint。

quote()按预测输入与显式条件匹配唯一价格档：

~~~text
cost = (预测输入×输入单价 + 预测输出×输出单价)/1_000_000 × 配置汇率
~~~

当前币种CNY、汇率1；C输入/输出0.56/1.89、D0.32/1.08元/百万tokens。无匹配、多档重叠、缺价/汇率不猜测，不假定缓存折扣或未来重试费用。D两项价格均低42.86%，会强烈影响当前结果。

### 6.2 性能是相似请求统计

| 指标 | 长度条件 | 默认统计 |
|---|---|---|
| E2E | 输入与输出均相似 | 各Endpoint近期加权均值，可选P50 |
| TTFT | 输入相似 | 同上 |
| TPOT | 输入相似 | 同上，不按输出拆分 |

δI=max(128,0.5×预测输入)，δO=max(50,0.25×预测输出)；区间[max(0,L−δ),L+δ]。这是双侧性能匹配，**不是SLO的单侧20%**。

~~~text
w_i = 2^(-完成后年龄分钟/180) × exp(-0.5×Σ((实际长度−预测长度)/容差)^2)
mean = Σ(w_i×指标_i)/Σw_i
n_eff = (Σw_i)^2/Σ(w_i^2)
~~~

mean是原单位典型延迟，不先除SLO，不是递推EWMA或单请求保证。statistic=mean,quantile=null可改为p50,0.5；加权P50由累计权重取中位数。

窗口60/180/360/720/1440分钟，满足40条且有效n≥20的最短窗口停止；否则用最大窗口。常规证据要求n≥30且有效n≥20；40决定扩窗、30决定等级。稀疏仍保留exploratory_estimate，当前策略明确读取作兜底，不将estimate=null伪装充分证据。

BP训练E2E与输出Pearson约0.978/Spearman0.944，与输入−0.165/0.143。输入处理可能影响总时延，但双长度条件是待验证假设，也会减少样本。TPOT输入有一定关联、输出较弱；不能把相关性当因果。

### 6.3 PerformanceIndex维护方式

| 结构 | Key/内容 | 用途 |
|---|---|---|
| samples | request_id→完整完成事实 | 活跃主记录只管理一份 |
| by_pair | (model,endpoint,stream_type)→ID集合 | 隔离组合和模式 |
| by_length | (model,endpoint,stream_type,input/output,桶号)→ID集合 | 输入桶128、输出桶256，加速筛选 |
| expiry | 完成时间顺序的ID双端队列 | TTL过期时同步清索引 |
| pending/position | 全部回放完成序列/当前位置 | 仅离线逐步导入历史 |
| as_of | 当前截止时间 | 对应当前到达，回放不可倒退 |

先对覆盖输入范围的桶取并集、输出桶取并集，再取交集，读主记录精确过滤边界/指标，最后按窗口统计。桶可能部分越界，不省略精确检查；索引只存ID不复制全部字段。128/256不是相似容差或SLO档位。

advance()只导入finished_at<arrived_at，超过24h同步移除主记录、组合和空桶。活跃缓存有时间界限，不是固定容量循环队列；pending仍持有全部离线事实，整个进程内存并非仅24h。不会逐请求重读磁盘。

当前还扫描组合窗口算mixed_window_p95作诊断，TTFT/TPOT分别估计；这些耗时已计入。混合P95不参与决策或补缺。共享匹配候选、关闭诊断是待优化项，不能宣称整个查询常数耗时。

## 7. 候选策略与返回契约

### 7.1 当前决策链

~~~text
RoutingEngine.route()
  → availability.route_available()
  → advance(到达时间)
  → NeighborhoodSLO.assign()：原始T，再独立计算A=1.2T
  → catalog.endpoints(model)
  → quote()/estimate()：逐Endpoint
  → 单项门槛与证据分层
  → rank_frontier(...,fill_dominated=True)
  → 空常规池时低证据/unknown兜底
  → decision、timing
~~~

明确禁用、价格错误、SLO缺失、已知指标超限不能通过兜底绕过。BP没有真实容量/健康/冷却检查，不能称其已通过。

~~~text
非流式loss = E2E估计/A_e2e
流式loss = η×TTFT估计/A_ttft+(1−η)×TPOT估计/A_tpot
score = λ×cost/median(可选候选正cost)+(1−λ)×loss
~~~

默认λ=0.5、η=0.5，score小为好。流式先独立检查两项，即使某项权重0也不跳过。归一化后才合成不同单位指标；没有正成本时参考值1，不混币种。0.5不保证实际贡献各50%，大SLO会压小性能损失及差距。

A支配B，当A成本/loss均不更大且至少一项更小。前沿按(score,endpoint_id)升序；前沿优先，不足3位从**同一合格池**按score补被支配点。补点pareto=false，不把前三位强称纯前沿；最多3个不同Endpoint，不足null。

### 7.2 低证据

常规池为空才启用：先用相似稀疏样本的相同均值/P50、已知指标必须逐项≤A，做前沿与补位；有余位且允许unknown时补无相似历史或缺部分指标的点。

unknown排在有值稀疏点之后，按已知指标数降序、费用、ID排序，不伪造性能0、score或前沿坐标；已知超限不绕过。常规池非空不为凑满混入低证据。allow_unknown_performance=true是best-effort许可，业务上线需单独批准。

| 字段 | 含义 |
|---|---|
| slots | 推荐顺序3位可含null，不实际执行 |
| frontier_ranked | 有已知数值点的前沿，不含unknown伪点 |
| selected | 常规候选存在，不等于HTTP成功 |
| selected_low_evidence | 使用兜底，风险必须展示 |
| no_feasible_endpoint | 无可选，不强行选非法点 |
| pareto_frontier / dominated_fill | 前沿选择 / 合格被支配补位 |
| sparse_evidence_fallback:… | 稀疏点前沿/补位 |
| unknown_performance_best_effort_fallback | 未知性能尝试 |
| risk_warnings | 如best_effort_no_slo_guarantee，不得丢弃 |
| feasible / eligible_for_selection | 前者常规通过，后者可含兜底，不混为认证 |

decision还包含T/A、价格、routing_estimates、n/有效n/窗口/来源ID/权重、loss/score；timing单独返回。既有稳定备用不属于本策略。

## 8. 当前已实现的最小离线接口

从BP目录运行，需现有结果包，不发HTTP：

~~~python
import json
from pathlib import Path
from bp_slo.performance import PerformanceIndex
from bp_slo.routing import ModelEndpointCatalog, RoutingEngine

root = Path("runs/bp_routing_relative20_20261008")
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

cfg = read_json(root / "config.json")
engine = RoutingEngine(
    read_json(root / "baseline/slo_rules.json"),
    read_json(root / "tolerance_config.json"),
    cfg,
    ModelEndpointCatalog(read_rows(root / "endpoint_offerings.jsonl")),
    PerformanceIndex(read_rows(root / "baseline/dataset/all_requests.jsonl"), cfg["performance"]),
)
decision, timing = engine.route(read_rows(root / "replay_requests.jsonl")[0])
print(decision["slots"], decision["status"])
~~~

复用同一个engine，后续到达必须非递减；不要每请求重建。加载完整完成事实只是离线方法，不是线上预知未来的设计。

## 9. 当前实验、结果和限制

### 9.1 结果版本

| 目录 | 用途 |
|---|---|
| runs/bp_baseline_20261007 | BP提取、事实、画像、配置 |
| runs/bp_neighborhood_slo_relative20_20261008 | **当前SLO**：20%邻域、冻结规则、覆盖 |
| runs/bp_routing_relative20_20261008 | **当前路由**：均值/1.2/补位和低证据 |
| runs/bp_neighborhood_slo_20261008 | 旧固定200邻域SLO，对照 |
| runs/bp_routing_available_20261008 | 固定200SLO＋新候选政策，对照 |
| runs/bp_routing_neighborhood_20261008 | 固定200＋P95/前沿-only，7条中4条有候选 |
| 其他固定分桶/优化归档 | 历史研究，不作当前默认 |

当前502条非流式测试中493条局部分位数、6条稀疏、3条缺预测全局参考。**1.1覆盖评估**：

| 指标 | strict | standard | relaxed | 可评估n |
|---|---:|---:|---:|---:|
| 非流式E2E | 78.09% | 95.82% | 97.81% | 502 |
| 流式TTFT | 98.62% | 99.08% | 99.08% | 218 |
| 流式TPOT | 93.12% | 100.00% | 100.00% | 218 |
| 流式两项同时满足 | 91.74% | 99.08% | 99.08% | 218 |

流式总265、成功232、失败33，218为指标有效的评估分母，不是全请求成功率。这张表不代表1.2路由政策实际达标。

### 9.2 回放设计与选择

767条测试输入排除17条预测不完整，非流式按预测输出分组、流式按预测输入分组，每组取按到达排序的中间一条，共7条，不看结果。缓存由完整原日志完成事件推进，不模拟新策略反馈。

| 请求 | 模式 | 预测输入/输出 | 证据 | 候选 |
|---|---|---:|---|---|
| req_011302 | 非流式 | 2878/654 | 常规 | D,C,null |
| req_011511 | 非流式 | 12700/2139 | 稀疏，C5/D1 | D,C,null |
| req_011618 | 非流式 | 3065/323 | 常规 | D,C,null |
| req_020642 | 流式 | 237/11922 | 常规 | D,C,null |
| req_020661 | 流式 | 122/17492 | 常规 | D,C,null |
| req_021704 | 流式 | 563/48864 | 常规 | D,C,null |
| req_021823 | 流式 | 2844/108790 | 稀疏，C22/D16 | D,C,null |

5常规、2低证据、0空候选，首选全D；4条常规C补位。同样7条旧P95版本有4条候选，但统计量/倍率/政策同时变，不单独归因某项，也不称真实成功率提升至7/7。

只在原历史真正执行Endpoint上，MAE为E2E3条1.418秒、TTFT4条0.655秒、TPOT4条5.489ms/token。历史C而新选D的请求没有D实际结果；不能做所选D的反事实收益结论。混合P95与均值目标不同，误差不能直接作为公平收益比较。

7条本机开销：状态平均1.568ms、查表0.114ms、性能/成本7.635ms、排名/兜底0.046ms；决策不含状态平均7.794ms/P95 11.250ms，总平均9.363ms/P95 13.257ms。timings.jsonl原始ns，timing_summary.json数值已转ms即使字段名仍_ns。包含调试诊断，截止排序，不含加载/绘图/写文件/最终返回对象封装，不是生产吞吐基准。

### 9.3 已知问题

- strict非流式是P50，不应自然期待95%。513–2048的425条带10%为74.59%/95.53%/97.88%；309条≤1000仍受200下限，20%不能改变多数取样。
- mean与历史P90/倍率不同口径。req_011511 SLO仅4条、P90约58.853秒，relaxed上限105.935秒；D仅1条29.100秒，两侧都弱。
- TTFT全局底线和尾部可让上限达均值十几倍，压低性能贡献。TPOT全局498不能证明Endpoint n22/16充分。
- 非流式超长无本轮实测覆盖，外推只有边界/合成测试；不报错不等于长作业承诺。
- 研发已观察旧测试期，需新的独立时间段；没有未选Endpoint真实结果、其他策略收益或HTTP闭环。

### 9.4 结果文件

SLO包含slo_rules.json、test_assignments/results、coverage、tolerant_results/coverage、assignment_methods、boundary_examples及图；边界探针不进覆盖分母。

路由包含replay_requests、sampling、decisions、observed_checks、summary、timings/timing_summary、每请求图、baseline冻结副本及配置。manifest记录源/产物/代码哈希，validation是生成时结果，应重跑。跨机器源不可达会警告，包内重算不等于验证原始提取完整性。

## 10. 融合完整系统的分步方案

以下是**接入建议，不是已实现的新API**。保留完整系统状态/执行框架，逐层迁移，勿整包照搬。

### 第一步：规则与单位冻结

交付代码、三个当前配置、对应slo_rules.json和验证材料；记录模型、规则/训练版本、代码哈希。启动缓存规则，更新时原子切换并可回滚，不手改旧结果。泛化BP硬编码校验与模型规则注册，不能只替换model_id字符串。

统一stream/nonstream、带时区时间、ms/ms-token、币种、预测/实际tokens、原始T/验收A。

### 第二步：上游预测与请求级SLO适配

上游给固定模型、模式、真实预测及版本，业务档位需明确契约。复用NeighborhoodSLO查询三档，由适配器选一档。

旧包request_routing.py有RequestSLO/RoutingRequest；需要适配arrived_at字符串→arrived_at datetime、stream_type→is_stream，旧from_prepared()使用的arrival_time/token_predictions也是另一格式。

| BP原始查询 | 完整系统字段 |
|---|---|
| grades[tier]['e2e']['limit'] | RequestSLO.e2e_ms |
| grades[tier]['ttft']['limit'] | RequestSLO.ttft_ms |
| grades[tier]['tpot']['limit'] | RequestSLO.tpot_ms，口径ms/token |
| 选定tier | RequestSLO.tier |

保留规则版本、来源、fallback/外推/冷启动标记。**不要把1.1/1.2后的值存为原始RequestSLO再重复放宽**，指定唯一准入层计算A。当前BP实例grade固定，需要显式支持请求选档。先影子查询核对，旧route_request()仍是旧性能/备用政策，换SLO并不等于启用新策略。

### 第三步：真实完成事件的索引

旧memory_state.py的InMemoryRoutingState.record_observation()可作为反馈挂点，但不是BP长度索引。新增适配器处理实际Endpoint、attempt起止、成功/失败、真实tokens、指标、质量；完成后去重入索引，不提前读标签。

旧models.py的Observation只有occurred_at、Endpoint、模式、结果和延迟等字段，**没有request_id、实际输入/输出tokens及独立起止时间**。不能直接把它当BP的完成事实，需要扩展事件契约或关联执行层元数据。旧窗口默认保留360分钟，BP最大窗口1440分钟；迁移时也要统一保留期，避免查询24小时实际只有6小时的数据。优先让一份完成事件派生质量/性能索引，而不是维护互相不一致的两套事实。

保留单主记录、组合/模式/长度索引与TTL，补锁/原子快照、乱序、内存/样本上限及持久日志恢复。当前PerformanceIndex只有离线pending推进，没有线上record_completion()接口，**不能随意pending.append就当在线已完成**；非递减到达限制不能直接用于多线程请求。迁移后使用真实事件，不预加载未来；正确处理重试归因，不把旧P95/先验当新均值。

### 第四步：硬准入、费用与候选政策

RPM/TPM/并发归属于(model,endpoint)，不是Endpoint跨模型总体。保留既有容量/启停/冷却/健康检查，在同一状态版本预占，避免多个决策共用同一空余。

统一价格档/货币，BP显式汇率与旧包同币种政策不可混用。引入长度相关均值/P50及T→A→loss，再显式替换旧稳定备用为前沿优先和合格被支配点补位；旧短列表和BP3位null也要适配。

低证据只允许放宽“样本数量不足”，不能绕过硬容量、禁用、冷却、已知超限或价格错误；unknown业务显式控制。先影子比较，再灰度。

### 第五步：执行与反馈闭环

执行层消费slots顺序；开始预占，完成/取消/超时均释放，真实tokens校正TPM，结果记真实Endpoint。重试前重查剩余预算、容量/健康和安全，流式已输出后不能盲目重试。BP只生成候选，不执行这些动作。

E2E/TTFT从网关到达计时，排队/重试扣剩余预算，不每次重新获得完整SLO。真实首token、TPOT测量需替换代理口径；新策略产生自己反馈后才能证明收益。

### 第六步：分层验收

| 层 | 必须验证 |
|---|---|
| 契约 | tokens/时间/单位/币种、档位、T与倍率一次、模型隔离 |
| SLO | 训练可见性、档位、稀疏/外推/冷启动，新期覆盖及预算宽度 |
| 索引 | 去重/乱序/过期/并发快照、无串用、与全量查询一致 |
| 选择 | 单项门槛、前沿/补位、空池兜底、未知不伪造、3位不重复 |
| 硬约束 | 原子预占、取消释放、未知容量、冷却/禁用，兜底不绕过 |
| 闭环 | 调用/重试/归因、流式安全、排队及剩余预算 |
| 收益/性能 | 后续同请求环境对照、失败分母、尾延迟/费用/吞吐/路由并发耗时 |

先证明本策略接口和安全，再按批准范围接常见策略比较；Endpoint模拟只支撑决策验证，不作为重点。繁忙攒批、重请求调度与数据库另排任务，不阻塞候选层融合。

## 11. 修改入口与交付清单

| 修改项 | 位置 | 需要重做 |
|---|---|---|
| SLO20%/200/样本数、TTFT、三档/冷启动 | slo_neighborhood.json、neighborhood_slo.py | 新SLO包、边界/覆盖、路由新baseline |
| mean/P50、相似范围、窗口/半衰期/证据 | routing_available.json.performance、performance.py | 误差、兜底、开销回放 |
| 1.2/补位/unknown | selection_policy、availability.py | 门槛、风险、排序测试 |
| λ/η、实例grade | routing_available.json | 同样本重跑；当前非请求级grade |
| 10%评估 | slo_tolerance.json、tolerance.py | 新覆盖报告，不改T、不重复路由 |
| 价格/币种/档位 | offering、quote() | 新来源配置，不手改manifest |
| 在线状态/容量/HTTP | 完整系统适配层 | 无BP现成功能，需单独实现验证 |

交付代码、测试、配置、requirements/pyproject和README；复现数值另带当前SLO/路由包及BP提取配置，或有权限的三份源数据。runs被.gitignore排除，Git checkout不默认有真实数值。不要意外上传流量、凭据或本机资料。

禁止：完成结果做当前特征；缺样本变失败/零耗时；全局SLO当Endpoint性能；低证据当充分保证；补位仍称纯前沿-only；实验覆盖或7/7候选当生产SLA/收益。融合必须保留版本、单位和风险解释。
