"""Chinese, data-driven report. Units are converted only for presentation."""
from datetime import timedelta, timezone
from pathlib import Path

from .dataset import timestamp


def fmt(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"] +
                      ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def stats_table(groups, unit, scale=1):
    def scaled(row, key):
        value = row[key]
        return fmt(None if value is None else value / (scale**2 if key == "variance" else scale))
    return table(["Tokens 范围", "n", f"均值（{unit}）", f"方差（{unit}）²", "标准差", "P50", "P75", "P90", "P95", "样本提示"],
                 [[r["range"], r["n"], *[scaled(r, k) for k in ("mean", "variance", "std", "p50", "p75", "p90", "p95")],
                   "初筛数量足够" if r["sample_status"] == "enough_for_initial_screen" else "稀疏/无样本"] for r in groups])


def build_report(analysis, rows, offerings, config):
    a, g, modes = analysis, analysis["groups"], analysis["modes"]
    names = {"nonstream": "非流式", "stream": "流式", "unknown": "类型未知"}
    dates = [timestamp(s).astimezone(timezone(timedelta(hours=8))).isoformat() for s in a["time_range_utc"]]
    split_ratios = "/".join(f"{v:.0%}" for v in (config['train_fraction'], config['calibration_fraction'], 1-config['train_fraction']-config['calibration_fraction']))
    estimated = [r for r in rows if 'token_counts_locally_estimated' in r['quality_flags']]
    sections = [f"# {a['model_id']} 独立数据集与样本分析\n\n"
                f"数据覆盖：{dates[0]} 至 {dates[1]}（UTC+8）。\n\n"
                "本报告只做样本审计和探索分析，**尚未定义或冻结 SLO 阈值**。所有 SLO 画像合并该模型的所有 Endpoint，"
                "不分别为 Endpoint 制定 SLO。数据来自已完成请求日志，不是原始到达请求，也不是反事实路由实验。"]
    sections.append("## 1. 样本与质量\n\n" + table(
        ["类型", "全部请求", "成功", "失败", "成功率", "输出长度 + E2E 有效", "输入长度 + TTFT 有效", "输出长度 + TPOT 有效"],
        [[names[k], m['total'], m['success'], m['failures'], f"{m['success_rate']:.2%}" if m['success_rate'] is not None else "—",
          m['length_e2e_eligible'], m['ttft_input_eligible'], m['tpot_output_eligible']] for k, m in modes.items()]) +
        "\n\n成功要求 `gateway_result=completed 且 http_status=200`。失败和类型未知的记录保留在数据集中；"
        "它们不混入成功时延分位数，不因失败而删掉整条原始记录。三个有效列是不同指标的独立口径，不能相加。\n\n" +
        table(["类型", "HTTP 状态分布", "成功但缺输入 tokens", "成功但缺输出 tokens"],
              [[names[k], "; ".join(f"{s}: {n}" for s, n in sorted(m['http_statuses'].items())), m['missing_input_success'], m['missing_output_success']] for k, m in modes.items()]) +
        f"\n\n原始 BP 记录有 {a['extraction']['source_order_arrival_reversals']} 处相邻到达时间倒序。`raw_requests.jsonl` 保持源记录顺序和字段值，"
        "`requests.jsonl` 按绝对到达时间排序，并保留原始文件 `source_line`。不是直接把源文件行序当成时间顺序。\n\n"
        f"预测字段可用的请求数：{a['raw_prediction_count']}。原始 token 数是完成后事实；本项目不把它们复制为预测值。"
        "没有提供预测时保留 null，并标记 `prediction_source=missing`。不凭空生成 SLO、优先级、轻重标签或紧急程度。\n\n" +
        table(["质量标记（可重叠）", "记录数"], sorted(a['quality_flags'].items())) +
        f"\n\n本地估算 tokens 标记涉及 {len(estimated)} 条，其中成功 {sum(r['success'] for r in estimated)} 条。"
        "本次这些估算记录不进入成功 SLO 画像；保留它们用于质量与错误分析。缺失指标的请求不一定与完整请求同分布，"
        "因此有效样本的分位数不能无条件推广到全部请求。")
    sections.append("## 2. 指标定义和统计口径\n\n"
        "- 非流式 E2E 使用请求级 `elapsed_ms`，包括网关从到达到完成的整体时延；保留并核对结束减开始的差值。\n"
        "- 流式 TTFT 使用最后一次匹配 Endpoint 的成功 attempt 中 `first_token_at_ms`。源说明明确它是**首响应延迟代理值**，"
        "虽然字段名带 `_at`，它不是要再减 `sent_at_ms` 的绝对时间戳。请求级首响应代理另存，不能混用时间范围。\n"
        "- 流式 TPOT 代理 = `(attempt.finished_at_ms − attempt.sent_at_ms − TTFT_proxy) / (attempt.output_tokens − 1)`。"
        "仅在 attempt 成功、完整计时有效、输出 tokens > 1 且时长合法时计算；attempt 未提供输出数时才回退到请求输出数。"
        "它是单请求的平均解码间隔，不是逐 token 间隔的 P95，也不能证明没有流式卡顿。\n"
        "- 分位数采用线性插值：排序后位置为 `(n−1)q`。方差为样本方差 `Σ(x−均值)²/(n−1)`；n<2 时方差为 null，不能写成 0。\n"
        "- 方差和标准差只用于观察分布离散程度，不参与 Pareto 排序。方差大不等于一定随时间不稳定；还要看长度构成和时间分布。\n"
        f"- 每组 {config['minimum_group_samples']} 条只是可配置的探索初筛门槛，不是显著性证明或 P95 可信度保证。n=100 时尾部大约只有 5 条，"
        "不能据此直接作业务承诺。缺失不当作 0，不插补真实时延，不删除合法长尾。\n"
        "- 统计按请求等权，不按 Endpoint 等权。因此画像反映历史路由混合比例，不代表所有 Endpoint 的潜在最优性能。")
    length_rows = []
    for mode in ("nonstream", "stream"):
        for key, label in (("actual_input_tokens", "输入"), ("actual_output_tokens", "输出")):
            m = modes[mode]["metrics"][key]
            length_rows.append([names[mode], label, m['n'], fmt(m['mean']), fmt(m['variance']), fmt(m['std']), fmt(m['p50']), fmt(m['p95']), fmt(m['max'])])
    overview = []
    for mode, metric, label, scale, unit in (
        ('nonstream', 'request_e2e_ms', '非流式请求 E2E', 1000, '秒'),
        ('stream', 'ttft_proxy_ms', '流式 TTFT 代理', 1, '毫秒'),
        ('stream', 'tpot_proxy_ms', '流式 TPOT 代理', 1, '毫秒/token'),
    ):
        m = modes[mode]['metrics'][metric]
        overview.append([label, unit, m['n'], fmt(m['mean']/scale) if m['mean'] is not None else '—',
                         fmt(m['variance']/scale**2) if m['variance'] is not None else '—',
                         *[fmt(m[k]/scale) if m[k] is not None else '—' for k in ('std', 'p50', 'p75', 'p90', 'p95')]])
    sections.append("## 3. 成功请求的总体指标与长度画像\n\n" +
        table(['指标', '单位（方差为单位平方）', 'n', '均值', '方差', '标准差', 'P50', 'P75', 'P90', 'P95'], overview) +
        "\n\n总体 E2E 可使用没有 token 长度的成功记录，故总体 n 与长度分桶 n 可能不同。\n\n" + table(
        ["类型", "Tokens", "有效 n", "均值", "方差", "标准差", "P50", "P95", "最大值"], length_rows) +
        "\n\n上述长度来自完成后记录。这里只用来认识数据；上线分桶必须使用到达时可获得的预测长度，并单独评估预测误差造成的错桶。")
    sections.append("## 4. 非流式：输出长度与 E2E\n\n" + stats_table(g['nonstream_e2e_by_output'], "秒", 1000) +
        "\n\n解释：短输出与长输出的 E2E 确实不同，输出长度分桶有数据依据。但分桶边界只是探索用的预设尺度，并非已学习出的最优边界。"
        "长输出桶明显稀疏，不能继续细分后把几个观测值当成可信的高分位数。需要补充独立长请求数据，或明确降级/合并策略。\n\n"
        "不能将 `E2E / output_tokens` 解释为恒定的每 token 生成时间：E2E 还包含首响应、输入处理、网络和可能的等待。"
        "输出长度相关不等于只有输出长度有影响。以下补充输入长度的边际分布，但没有控制输出长度，不能解释为输入长度的因果效应。\n\n" +
        stats_table(g['nonstream_e2e_by_input'], "秒", 1000))
    sections.append("## 5. 流式：输入长度与 TTFT 代理\n\n" + stats_table(g['stream_ttft_by_input'], "毫秒") +
        "\n\nTTFT 的输入长度分组存在差异，值得作为后续条件 SLO 的候选变量。但大输入组样本稀疏，不能对现有输入范围之外外推。"
        "输入 tokens 不是输入处理工作的全部信息，缓存命中、队列和历史路由比例也会影响结果。"
        f"成功有效 TTFT 代理的最大值为 {fmt(modes['stream']['metrics']['ttft_proxy_ms']['max'])} ms；保留合法长尾，不为了缩小方差将其删除。")
    sections.append("## 6. 流式：TPOT 是否需要长度分桶\n\n### 按输出长度\n\n" + stats_table(g['stream_tpot_by_output'], "毫秒/token") +
        "\n\n### 按输入长度（辅助观察）\n\n" + stats_table(g['stream_tpot_by_input'], "毫秒/token") +
        "\n\n目前中长输出各组 TPOT 的高分位数较接近，相比 E2E，按输出长度分桶的必要性弱一些。"
        "可以把“模型统一 TPOT 阈值”作为后续待验证的简洁基线，而非现在就认定所有长度完全一致。"
        "短输出组样本不足；输入较长的 TPOT 组也稀疏，需避免把路由、长度和时间构成差异误当成规律。"
        "TPOT 直方图还显示多个密集区，不能用单一均值掩盖混合分布；这不改变按模型合并设计 SLO 的约定，但需要在后续覆盖验证中关注。")
    assoc_names = {"nonstream_output_e2e": "非流式：输出 tokens ↔ E2E", "nonstream_input_e2e": "非流式：输入 tokens ↔ E2E",
                   "stream_input_ttft": "流式：输入 tokens ↔ TTFT", "stream_output_tpot": "流式：输出 tokens ↔ TPOT", "stream_input_tpot": "流式：输入 tokens ↔ TPOT"}
    sections.append("## 7. 关联与时间稳定性\n\n" + table(["关联", "有效配对 n", "Pearson", "Spearman"],
        [[assoc_names[k], v['n'], fmt(v['pearson']), fmt(v['spearman'])] for k, v in a['correlations'].items()]) +
        "\n\nPearson 描述线性关联，Spearman 描述秩的单调关联；都不是因果结论，也不能单凭相关系数决定 SLO。\n\n"
        f"按每种模式分别按到达时间尝试 {split_ratios} 切分，仅作为诊断。相同到达时刻不会被拆开；"
        "训练时还必须排除在训练截止后才完成的请求（见 `training_outcome_visible`）。此切分没有被批准用于正式校准。\n\n" +
        table(["模式", "区间", "请求数", "输出 + E2E 有效数", "输出桶分布"],
              [[names[v['mode']], v['split'], v['total'], v['valid'], "; ".join(f"{b}: {n}" for b, n in v['output_bins'].items())] for v in a['split_coverage']]) +
        "\n\n**关键限制：时间段间长度构成变化很大。非流式短请求的校准样本尤其不足，简单的按模型划分时间段也不能自动解决覆盖问题。**"
        "不要为凑数随机打乱时间后声称能验证线上效果。后续需补充独立时间段，或在有足够覆盖的滚动时间折上验证，"
        "每折只使用当时已经完成的数据。全量画像可以帮助发现问题，但正式选阈值不能再反复使用测试集。\n\n"
        "`analysis.json` 另含按小时的样本数、均值、方差和分位数；当前仅约一天的数据，无法据此证明跨天、周末或版本更新后的稳定性。\n\n"
        "![长度与时延](figures/distributions.png)\n\n![时间切分覆盖](figures/temporal_coverage.png)")
    price_rows = []
    for ep in offerings:
        price = ep['price_config'] or {}
        for tier in price.get('price_tiers', []):
            price_rows.append([ep['endpoint_id'], price.get('currency'), price.get('billing_unit'),
                               f"({tier.get('input_tokens_min_exclusive')}, {tier.get('input_tokens_max_inclusive')}]",
                               tier.get('input_per_million'), tier.get('output_per_million'), tier.get('cache_read_per_million')])
    sections.append("## 8. Endpoint 与后续 Pareto 路由\n\n" + table(
        ["Endpoint", "模式", "请求", "成功", "输出 + E2E 有效", "输入 + TTFT 有效", "输出 + TPOT 有效"],
        [[r['endpoint_id'], names[r['mode']], r['total'], r['success'], r['length_e2e_eligible'], r['ttft_input_eligible'], r['tpot_output_eligible']] for r in a['endpoint_counts']]) +
        "\n\n这里只列覆盖与质量，不按 Endpoint 分别设置 SLO。报价从原配置原样保留：\n\n" +
        table(["Endpoint", "货币", "计价单位", "输入档位", "输入价", "输出价", "缓存读价"], price_rows) +
        "\n\nBP 的 C、D 都只有一个价格档，均为 CNY/百万 tokens；同一长度下 C 的单价是 D 的 1.75 倍。"
        "容量 RPM、TPM、并发上限均未配置，不能把未知写成 0，也不能据此判断繁忙。\n\n"
        "新项目已独立实现并测试 `pareto.rank_frontier()` 的排序约定：先取可选 Endpoint 的非支配集合，再按"
        " `score = λ × (预计请求成本 / 成本参考值) + (1−λ) × 已归一化性能损失` 升序排列；相同评分用 Endpoint ID 确定顺序。"
        "参考值取可选候选的正成本中位数，全零成本时用 1。性能和成本越小越好；λ 越大越偏好低成本。\n\n"
        "最多取三个位置，不足写为 null。**不按稳定性重排备用端点，不拿被支配端点补足，不取第二层 Pareto 前沿。**"
        "BP 当前仅 C、D 两个 Endpoint，最多两个非空；若一个支配另一个，则只剩一个。纯排序函数不声称已完成在线路由。\n\n"
        "待 SLO 和到达时预测字段确定后，再把预计 token 成本及符合指标口径的 Endpoint 性能接入函数。"
        "目前没有真实流量回放选路、SLO 达标率评估或重试执行器。大多数请求只有一个 attempt，"
        "同一请求在另一个 Endpoint 的结果未被观察，不能把改选 Endpoint 的收益当成已经得到实证。")
    extraction = a['extraction']
    sections.append("## 9. 历史快照怎么使用\n\n"
        f"保留 BP 请求引用的 {extraction['unique_snapshot_count']:,} 个去重快照，范围计数：`{extraction['snapshot_scopes']}`。\n\n"
        "`endpoint_model` 是 BP 的端点历史；`model_all_endpoints` 是 BP 的合并历史；`endpoint` 是端点全模型背景，"
        "其中包含其他模型的信息，仅为引用完整性保留，不能当作 BP 逐请求样本。\n\n"
        "3 天、7 天窗口相互重叠，同一快照也会被不同请求引用，**不能累计快照计数扩充独立样本量**。"
        "这些聚合未提供足够的流式/非流式 × 长度条件明细，也无法从均值恢复方差或 TPOT 分位数。"
        "故本报告所有核心 SLO 统计来自逐请求事实，不拿快照补长请求样本。\n\n"
        "验证器检查引用闭合、模型/端点/窗口范围和 as-of 时间不晚于请求；但 as-of 合法只表明事件时间，"
        "不等于已证明生产环境当时没有回填或迟到数据。")
    sections.append("## 10. 验证、结论与下一步\n\n"
        "运行目录的 `manifest.json` 记录源文件及产物 SHA-256，`validation.json` 记录验证结果。"
        "验证会重新归一化原始 BP 记录、重新生成统计和切分，检查数量、唯一性、排序、引用范围、时间与文件完整性。"
        "SHA-256 用于发现意外修改，不是独立可信签名。原始来源不可访问时，只能完成包内验证，会明确报告警告。\n\n"
        "1. BP 可作为这次独立探索的模型，非流式总量并非完全不足，但长输出和部分时间段的样本明显不足。\n"
        "2. 非流式先验证“预测输出长度 → 长度区间 → 条件 E2E 分位数”；流式先验证“输入长度 → TTFT 分位数”，TPOT 先保留统一阈值候选。\n"
        "3. 不把 tokens 在桶内的位置百分比直接当作延迟分位数。二者没有必然对应关系；若以后插值，需证明单调、边界连续并在校准集检查覆盖。\n"
        "4. 下一阶段先确定可评估区间、补足预测字段与时间外验证样本，再制定可解释的阈值候选。业务可接受时限与历史可达能力要分别说明，"
        "历史 P95 不自动等于业务 SLA，也不能用测试数据反复调阈值。\n"
        "5. 之后才把每请求 SLO 接入 Pareto 前沿前三候选，检验规则执行与路由收益；方差只作画像，备用端点不再稳定性优先。\n\n"
        "当前完成的是独立、可追溯的 BP 数据包与详细画像；未训练 token 预测器、未正式制定 SLO、未实现繁忙/攒批/调度或反事实收益验证。")
    return "\n\n".join(sections) + "\n"


DATASET_GUIDE = """# 独立 BP 数据包

这不是在线输入请求数据集，而是已完成请求的观测事实及其历史引用。SLO 尚未标注。

| 文件 | 作用 |
| --- | --- |
| raw_requests.jsonl | 仅 BP 源请求，保留原字段值、attempts 与历史引用，源顺序不变（JSON 排版可能不同） |
| requests.jsonl | 排序后的统一请求事实；用于主要统计 |
| endpoint_offerings.jsonl | BP 在 C/D 的原始价格档、容量字段及源模型配置 |
| historical_references.jsonl | request → snapshot 的引用、scope、model、endpoint、窗口 |
| historical_snapshots.jsonl | 被引用快照的闭包；endpoint scope 是全模型背景，不是 BP 额外样本 |
| diagnostic_splits.jsonl | 按模式时间切分的覆盖诊断，不是已冻结的正式训练/校准/测试集 |

## requests.jsonl 字段组

| 字段 | 含义与可用时间 |
| --- | --- |
| schema_version, record_kind | 版本与 completed_request_fact 类型；防止误当成到达请求 |
| request_id, model_id, user_id, stream_type | 源身份与请求模式；未知模式保留 unknown |
| source_line | 这条记录在原始全模型流量文件中的行号 |
| arrived_at, finished_at | 带时区的绝对时间，统一存 UTC；结束时间由相对事件时间恢复 |
| success, gateway_result, http_status, final_endpoint_id | 完成后的结果及实际端点；不能泄漏到到达请求特征 |
| actual_input_tokens, actual_output_tokens, actual_cache_hit_tokens | 完成后的观测 token 数，缺失为 null；源估算标记另保留 |
| predicted_input_tokens, predicted_output_tokens, prediction_source | 仅接受源记录已提供预测；没有则 null/missing，不回填实际值 |
| request_e2e_ms, request_ttft_proxy_ms | 请求级 E2E 与首响应代理；均为完成后标签 |
| attempt_e2e_ms, ttft_proxy_ms, tpot_proxy_ms | 匹配最终 Endpoint 的成功 attempt 指标；TTFT 是代理，TPOT 是派生均值 |
| final_attempt_output_tokens | 计算 TPOT 的分母长度，不把 N≤1 的请求除以零 |
| attempt_count, source_attempt_count | 实际 attempts 数与源字段；完整 attempts 在 raw 中 |
| source_usable_for_token_workload, source_usable_for_full_attempt_timing | 保留源数据可用性标记，不宣称人工标记就是真值 |
| eligible_length_e2e, eligible_ttft_input, eligible_tpot_output | 各项统计的独立有效性条件，不是统一“一刀切”清洗标记 |
| quality_flags, metric_scope | 质量原因与时间范围说明；保留失败/缺失/未知模式 |
| historical_snapshot_ids | 去重引用 ID；具体引用范围见 historical_references |

## 拆分字段

`split` 是 train/calibration/test/unassigned；`training_outcome_visible` 表示训练区间的请求是否在训练截止前已完成。
只按到达时间属于训练区间，不足以证明其完成后标签当时可用。
当前切分有严重的长度覆盖不均，仅用于诊断；不要不经检查直接训练或调 SLO。

离线画像允许使用 actual_* 和结果标签；后续构建在线 RequestInput 时，只能传 model、stream、到达信息、预测长度和外部业务要求等当时可得字段。
不得传入实际输出长度、最终 Endpoint、真实耗时和请求自己的完成结果。
"""


def write_reports(run_dir, analysis, rows, offerings, config):
    root = Path(run_dir)
    (root / "REPORT.md").write_text(build_report(analysis, rows, offerings, config), encoding="utf-8")
    (root / "dataset" / "README.md").write_text(DATASET_GUIDE, encoding="utf-8")
