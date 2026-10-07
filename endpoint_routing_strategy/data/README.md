# 初始请求数据处理

## 数据目标与边界

原始流量日志记录请求执行结束后的事实。目标数据表示请求刚到达网关、还没有选择 Endpoint、还没有执行时的输入。

历史日志中可以提取 request_id、user_id、model_id、到达时间和 stream_type。原始 messages、prompt 和请求到达时的真实 Token 预测没有记录，无法完整恢复原始 HTTP 请求正文。本阶段构造的是可用于元数据路由实验的初始请求；缺失预测均有 mock 来源标记。

初始请求禁止包含实际输入/输出 Token、最终 Endpoint、HTTP 状态码、响应耗时、执行结束时间、attempt、成功失败和执行后质量标记。

## 数据文件

| 文件                          | 数据阶段       | 用途                                       |
| --------------------------- | ---------- | ---------------------------------------- |
| incoming_requests.jsonl     | 请求刚到达      | 全部有效请求的初始输入元数据，供未来回放入口读取                 |
| requests.jsonl              | 网关画像完成     | 校准/测试区间的初始请求补充优先级、SLO、轻重、紧急程度后的调度输入      |
| canonical_requests.jsonl    | 执行后的规范事实   | 保留实际 Token、Endpoint、attempt 等，供历史校准和结果核对 |
| observations.jsonl          | attempt 完成 | 历史性能观察；只在 finished_at 之后可进入性能窗口          |
| profiles.json               | 离线历史先验     | 训练截止前已完成样本的中位数、P75及重型阈值                  |
| audit.json                  | 数据审计       | 缺失率、质量问题、预测来源、标签分布与计数                    |
| rejections.jsonl            | 清洗审计       | 原始行拒绝原因                                  |
| enrichment_rejections.jsonl | 规则审计       | 优先级与SLO冲突等画像拒绝原因                         |
| manifest.json               | 可复现信息      | 原文件哈希、产物哈希、时间边界及局限                       |
| resolved_config.json        | 实际配置       | 本次规则和随机种子                                |
| validation.json             | 验证结果       | 是否通过、错误分类和检查项目                           |
| summary.md                  | 阅读入口       | 本次处理概况                                   |

初始请求与执行后事实以 request_id 关联，路由算法只读取 incoming_requests.jsonl 或经过画像的 requests.jsonl。

## 初始请求字段

```json
{
  "schema_version": "1.0",
  "request_id": "req_example",
  "user_id": "user_example",
  "model_id": "model_example",
  "arrived_at": "2026-09-22T01:00:00+00:00",
  "is_stream": true,
  "predicted_input_tokens": 1024,
  "predicted_output_tokens": 256,
  "input_prediction_source": "mock_training_group",
  "output_prediction_source": "mock_training_group",
  "task_type": null,
  "priority": null,
  "slo_tier": null,
  "split": "test",
  "source_line": 100,
  "reconstruction": "arrival_metadata_from_completion_log",
  "payload_available": false
}
```

这是结构示例。实际模型代号、时间和预测值以生成数据为准。时间统一为 UTC；与原日志 UTC+8 是同一时刻。

priority 和 slo_tier 在初始元数据中可为空；网关画像阶段才按模拟业务规则补齐。字段如果由真实请求提供，则保留并检查一致性。

## Token预测处理

支持请求显式提供以下任一种格式：

```json
{"predicted_input_tokens": 1024, "predicted_output_tokens": 256}
```

或：

```json
{"token_predictions": {"input_tokens": 1024, "output_tokens": 256}}
```

已提供预测优先保留，0 是合法值，负数、非整数、布尔值和非有限数值标记为无效。

旧日志缺少预测时：

1. 训练/预热区间使用配置中的固定 mock，不使用这个区间未来结果。
2. 校准/测试区间使用训练截止前已完成样本的模型与流式类型中位数。
3. 组合样本少于 minimum_group_samples 时回退到训练全局统计；全局也不足时使用配置值。
4. 不把同一请求的执行后 input_tokens 或 output_tokens 改名为预测。

这种 mock 是流程输入占位，不证明真实预测器的准确性。历史中位数可能导致大部分请求为轻请求，这是本阶段的正常结果；后续按具体策略场景构造预测长度分布。

## SLO、优先级、轻重和紧急程度

配置文件为 experiments/configs/data_preparation.json。

### 三级优先级与三级SLO

| 优先级    | SLO等级    | 预计服务时间倍率 | 队列预算比例 |
| ------ | -------- | --------:| ------:|
| high   | strict   | 1.5      | 10%    |
| normal | standard | 2.0      | 20%    |
| low    | relaxed  | 3.0      | 30%    |

优先级缺失时，以 request_id 和 seed 的稳定哈希生成，默认比例为10%、70%、20%。它表达模拟业务重要性，与Token轻重独立。

SLO阈值根据模型/流式类型的冻结历史先验及预测工作量生成。训练统计仅作为模拟阈值基线，不是业务承诺。对于同一请求，strict、standard、relaxed的预算依次增大。

非流式预计服务时间：

```text
input_ratio  = predicted_input / max(training_median_input, 1)
output_ratio = predicted_output / max(training_median_output, 1)
estimated_service = training_P75_E2E × max(0.25, 0.3×input_ratio + 0.7×output_ratio)
SLO_E2E = estimated_service × tier_multiplier
```

流式预计服务时间：

```text
estimated_TTFT = training_P75_TTFT × sqrt(max(0.25, input_ratio))
estimated_service = estimated_TTFT + max(predicted_output - 1, 0) × training_P75_TPOT
SLO_TTFT = estimated_TTFT × tier_multiplier
SLO_TPOT = training_P75_TPOT × tier_multiplier
SLO_E2E = estimated_service × tier_multiplier
```

该服务时间模型是可替换的实验近似。默认系数用于合理初始化，尚未验证为真实Endpoint预测模型。

deadline 等于到达时间加 E2E 预算。流式另外维护 TTFT deadline。最大排队预算同时受 E2E 和 TTFT 约束，不能等待到首Token SLO已经不可能实现。

### 轻重请求

- 输入重型阈值：max(4096, 训练输入Token的P90)。
- 输出重型阈值：max(512, 训练输出Token的P90)。
- 判断使用到达时的预测Token，不使用实际输出。
- 两者均满足：mixed_heavy；仅输入：input_heavy；仅输出：output_heavy；均不满足：light。
- classification_heavy目前为null，task_type缺失为unknown。仅有Token和执行元数据无法可靠判断分类重型，不伪造标签。

轻重可以影响工作量和SLO预算，但不会自动改变业务优先级。

### 紧急程度

紧急程度是动态时间压力，不等于固定优先级：

```text
remaining_budget = deadline - now
slack = remaining_budget - estimated_remaining_service
slack_fraction = slack / total_budget
```

默认：slack_fraction不高于10%为critical，不高于30%为elevated，否则normal。流式还计算TTFT余量，使用更紧的一项。

初始时刻的大部分请求可以都是normal，不为制造特殊情况而随机赋予urgent。后续请求等待时用 urgency_at 重新计算。首Token已收到后，不再传入 estimated_remaining_ttft_ms，避免继续检查已达成的TTFT约束。

## 数据切分与防泄漏

按到达时间划分训练60%、校准20%、测试20%，相同到达时间不拆到不同区间。阈值和模拟预测只使用训练区间且在训练截止前已完成的结果。

incoming_requests.jsonl覆盖全部有效请求。requests.jsonl仅覆盖校准与测试请求，训练区间用于先验和预热，不拿冻结后的训练统计反向评价训练请求。

源日志first_token_at_ms是延迟代理而非绝对时刻；规范化层保留这一语义。缺失Token保持null，未完成attempt不会在到达或发送时发布最终性能。

## 项目结构

```text
data/
  schema.py       初始输入与执行后事实的数据对象
  normalize.py    唯一原始日志解析器
  profiles.py     训练先验、初始请求构造、画像规则
  pipeline.py     清洗编排、输出与审计
  validation.py   独立产物校验与重计算
  __main__.py     prepare / validate命令
tests/
  test_data_preparation.py
  test_router_strategy.py
```

旧路由演示和历史回放已共用normalize.py。project根目录pyproject.toml保存包元数据，requirements.txt记录运行依赖；当前实现无需额外安装依赖。

## 执行与验证

在项目根目录执行，自动创建新结果目录并验证：

```powershell
python -m endpoint_routing_strategy.data prepare
```

指定配置和输出位置（输出目录需要为空或不存在）：

```powershell
python -m endpoint_routing_strategy.data prepare --traffic .\历史性能数据包\流量记录_历史性能.jsonl --config .\endpoint_routing_strategy\experiments\configs\data_preparation.json --output-dir .\endpoint_routing_strategy\experiments\output\incoming_data_run1
```

独立重验已有产物：

```powershell
python -m endpoint_routing_strategy.data validate --dataset-dir .\endpoint_routing_strategy\experiments\output\incoming_data_run1
```

执行数据契约和原路由回归测试：

```powershell
python -m unittest discover -s .\endpoint_routing_strategy\tests -v
```

验证内容包括数量守恒、重复、来源、文件哈希、输入字段白名单、训练完成时间、预测保留、训练先验重算、画像重算、优先级与SLO映射、deadline及排队预算、轻重规则和完成后可见性。验证失败时命令返回非零退出码。

本步骤只完成普通请求数据准备，不注入故障、突发、繁忙或特殊重型分布。后续场景实验以这份输入为基础，保存独立场景配置和派生数据。
