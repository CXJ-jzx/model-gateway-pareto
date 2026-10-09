"""Pooled-model descriptive figures; no endpoint-specific SLO or fitted curve."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter


def make_plots(rows, analysis, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    specs = [
        (axes[0, 0], "nonstream", "actual_output_tokens", "request_e2e_ms", 1000,
         "Non-stream: output length vs request E2E", "Actual output tokens (log)", "Request E2E (seconds, log)", True),
        (axes[0, 1], "stream", "actual_input_tokens", "ttft_proxy_ms", 1,
         "Stream: input length vs first-response proxy", "Actual input tokens (log)", "Final-attempt TTFT proxy (ms, log)", True),
        (axes[1, 0], "stream", "actual_output_tokens", "tpot_proxy_ms", 1,
         "Stream: output length vs average TPOT proxy", "Actual output tokens (log)", "Final-attempt TPOT proxy (ms/token)", False),
    ]
    for ax, mode, x, y, scale, title, xlabel, ylabel, log_y in specs:
        items = [r for r in rows if r["success"] and r["stream_type"] == mode and r.get(x) is not None and
                 r.get(y) is not None and r[x] > 0 and (not log_y or r[y] > 0)]
        ax.scatter([r[x] for r in items], [r[y]/scale for r in items], s=12, alpha=.38, color="#227c9d", linewidths=0)
        ax.set(xscale="log", title=f"{title}\n(n={len(items):,}; endpoints pooled)", xlabel=xlabel, ylabel=ylabel)
        # Sparse log ticks stay readable for the narrow stream-input range.
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(float('inf'), float('inf'))))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=.18)
    values = [r['tpot_proxy_ms'] for r in rows if r['success'] and r['stream_type'] == 'stream' and r['tpot_proxy_ms'] is not None]
    axes[1, 1].hist(values, bins=35, color="#59a14f", alpha=.8, edgecolor="white")
    axes[1, 1].set(title=f"Distribution of per-request average TPOT\n(n={len(values):,}; not per-token jitter)",
                   xlabel="Final-attempt TPOT proxy (ms/token)", ylabel="Requests")
    axes[1, 1].grid(axis="y", alpha=.18)
    fig.suptitle("BP sample profile | completed-request observations, no SLO thresholds", fontsize=15)
    path1 = output_dir / "distributions.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, mode, field, group, title in (
        (axes[0], "nonstream", "output_bins", "nonstream_e2e_by_output", "Non-stream: output length + E2E"),
        (axes[1], "stream", "input_bins_ttft", "stream_ttft_by_input", "Stream: input length + TTFT proxy"),
    ):
        labels = [r['range'] for r in analysis['groups'][group]]
        portions = {r['split']: r for r in analysis['split_coverage'] if r['mode'] == mode}
        data = [[portions[s][field].get(label, 0) for s in ('train', 'calibration', 'test')] for label in labels]
        ax.imshow(data, cmap="Blues", aspect="auto", vmin=0)
        max_value = max((v for row in data for v in row), default=1)
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > max_value*.55 else "#1c3444")
        ax.set_xticks(range(3), ['Train', 'Calibration', 'Test'])
        ax.set_yticks(range(len(labels)), labels)
        ax.set(title=title, xlabel="Diagnostic time interval", ylabel="Actual tokens")
    fig.suptitle("BP temporal coverage | not approved as frozen calibration/test splits", fontsize=14)
    path2 = output_dir / "temporal_coverage.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    return [path1, path2]
