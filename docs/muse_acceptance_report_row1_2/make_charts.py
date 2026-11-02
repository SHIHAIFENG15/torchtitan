#!/usr/bin/env python3
"""Generate SVG loss / error / linearity charts without matplotlib."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "muse_report_data"
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def load_curve(path: Path) -> tuple[list[int], list[float], list[float]]:
    steps, losses, grads = [], [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        steps.append(int(parts[0]))
        losses.append(float(parts[1]))
        grads.append(float(parts[2]))
    return steps, losses, grads


def svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def polyline(xs, ys, x0, y0, w, h, xmin, xmax, ymin, ymax, stroke, width=1.5, opacity=1.0) -> str:
    pts = []
    for x, y in zip(xs, ys):
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        pts.append(f"{px:.2f},{py:.2f}")
    return (
        f'<polyline fill="none" stroke="{stroke}" stroke-width="{width}" '
        f'opacity="{opacity}" points="{" ".join(pts)}" />'
    )


def axes(x0, y0, w, h, xmin, xmax, ymin, ymax, xlabel, ylabel) -> str:
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#ccc"/>',
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#333"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#333"/>',
        f'<text x="{x0+w/2}" y="{y0+h+28}" text-anchor="middle" font-size="12">{svg_escape(xlabel)}</text>',
        f'<text x="18" y="{y0+h/2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 18 {y0+h/2})">{svg_escape(ylabel)}</text>',
    ]
    # y ticks
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        py = y0 + h - (yv - ymin) / (ymax - ymin) * h
        parts.append(f'<line x1="{x0-4}" y1="{py:.1f}" x2="{x0}" y2="{py:.1f}" stroke="#333"/>')
        parts.append(
            f'<text x="{x0-8}" y="{py+4:.1f}" text-anchor="end" font-size="10">{yv:.3g}</text>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+w}" y2="{py:.1f}" stroke="#eee"/>'
        )
    # x ticks
    for i in range(5):
        xv = xmin + (xmax - xmin) * i / 4
        px = x0 + (xv - xmin) / (xmax - xmin) * w
        parts.append(f'<line x1="{px:.1f}" y1="{y0+h}" x2="{px:.1f}" y2="{y0+h+4}" stroke="#333"/>')
        parts.append(
            f'<text x="{px:.1f}" y="{y0+h+16}" text-anchor="middle" font-size="10">{xv:.0f}</text>'
        )
    return "\n".join(parts)


def make_loss_panel(title: str, gpu_path: Path, npu_path: Path, out: Path) -> dict:
    gs, gl, _ = load_curve(gpu_path)
    ns, nl, _ = load_curve(npu_path)
    n = min(len(gl), len(nl), len(gs), len(ns))
    gs, gl, ns, nl = gs[:n], gl[:n], ns[:n], nl[:n]
    diff = [a - b for a, b in zip(nl, gl)]  # NPU - GPU
    abs_err = [abs(d) for d in diff]
    rel = [abs(d) / max(abs(b), 1e-12) for d, b in zip(diff, gl)]
    mae = sum(abs_err) / n
    mre = sum(rel) / n

    W, H = 920, 640
    x0, y0, w, h = 70, 50, 780, 240
    x1, y1 = 70, 360
    ymin = min(min(gl), min(nl))
    ymax = max(max(gl), max(nl))
    pad = (ymax - ymin) * 0.05 or 0.1
    ymin, ymax = ymin - pad, ymax + pad
    dmin, dmax = min(diff), max(diff)
    dpad = (dmax - dmin) * 0.1 or 0.01
    dmin, dmax = dmin - dpad, dmax + dpad

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        f'<text x="{W/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold">'
        f"{svg_escape(title)}</text>",
        axes(x0, y0, w, h, 1, n, ymin, ymax, "step", "loss"),
        polyline(gs, gl, x0, y0, w, h, 1, n, ymin, ymax, "#1f77b4", 1.6),
        polyline(ns, nl, x0, y0, w, h, 1, n, ymin, ymax, "#ff7f0e", 1.6, 0.9),
        f'<text x="{x0+10}" y="{y0+18}" font-size="12" fill="#1f77b4">GPU</text>',
        f'<text x="{x0+60}" y="{y0+18}" font-size="12" fill="#ff7f0e">NPU</text>',
        axes(x1, y1, w, h, 1, n, dmin, dmax, "step", "NPU − GPU loss"),
        polyline(gs, diff, x1, y1, w, h, 1, n, dmin, dmax, "#d62728", 1.4),
        f'<line x1="{x1}" y1="{y1 + h - (0 - dmin)/(dmax-dmin)*h:.1f}" '
        f'x2="{x1+w}" y2="{y1 + h - (0 - dmin)/(dmax-dmin)*h:.1f}" '
        f'stroke="#999" stroke-dasharray="4 3"/>',
        f'<text x="{x1+10}" y="{y1+18}" font-size="12">'
        f"MAE={mae:.5f}  MRE={mre*100:.2f}%</text>",
        "</svg>",
    ]
    out.write_text("\n".join(body))
    return {"mae": mae, "mre": mre, "n": n, "step1_gpu": gl[0], "step1_npu": nl[0],
            "step1000_gpu": gl[-1], "step1000_npu": nl[-1], "max_abs": max(abs_err)}


def make_perf_chart(out: Path) -> None:
    # Row1 both sides; Row2 NPU-only
    rows = [
        ("1 debugmodel", 322.89, 307.98, 1003.27, 924.81),
        ("2① with_cp", 148.53, 144.21, None, None),
        ("2② fsdp2", 121.64, 116.24, None, None),
        ("2③ mm_fsdp2", 47.94, 38.86, None, None),
    ]
    W, H = 960, 420
    x0, y0, w, h = 80, 60, 820, 300
    # plot linearity ratios as bars
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
            f'<text x="{W/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold">'
            "8卡/4卡 线性度（step&gt;100 稳定段）</text>",
            f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#ccc"/>']
    # y from 0.75 to 1.0
    ymin, ymax = 0.75, 1.02
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        py = y0 + h - (yv - ymin) / (ymax - ymin) * h
        body.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+w}" y2="{py:.1f}" stroke="#eee"/>')
        body.append(f'<text x="{x0-8}" y="{py+4:.1f}" text-anchor="end" font-size="10">{yv:.2f}</text>')
    # threshold line at relative: NPU vs GPU for row1 shown; 0.98 reference optional
    n_groups = len(rows)
    group_w = w / n_groups
    for gi, (name, n4, n8, g4, g8) in enumerate(rows):
        cx = x0 + group_w * (gi + 0.5)
        n_ratio = n8 / n4
        bw = group_w * 0.28
        # NPU bar
        nh = (n_ratio - ymin) / (ymax - ymin) * h
        body.append(
            f'<rect x="{cx-bw-4:.1f}" y="{y0+h-nh:.1f}" width="{bw:.1f}" height="{nh:.1f}" '
            f'fill="#ff7f0e"/>'
        )
        body.append(
            f'<text x="{cx-bw/2-4:.1f}" y="{y0+h-nh-6:.1f}" text-anchor="middle" font-size="10">'
            f"{n_ratio:.4f}</text>"
        )
        if g4 is not None and g8 is not None:
            g_ratio = g8 / g4
            gh = (g_ratio - ymin) / (ymax - ymin) * h
            body.append(
                f'<rect x="{cx+4:.1f}" y="{y0+h-gh:.1f}" width="{bw:.1f}" height="{gh:.1f}" '
                f'fill="#1f77b4"/>'
            )
            body.append(
                f'<text x="{cx+4+bw/2:.1f}" y="{y0+h-gh-6:.1f}" text-anchor="middle" font-size="10">'
                f"{g_ratio:.4f}</text>"
            )
        else:
            body.append(
                f'<text x="{cx+bw/2:.1f}" y="{y0+h-20:.1f}" text-anchor="middle" font-size="10" fill="#888">'
                "GPU 待测</text>"
            )
        body.append(
            f'<text x="{cx:.1f}" y="{y0+h+20}" text-anchor="middle" font-size="11">{svg_escape(name)}</text>'
        )
    body.append(f'<text x="{x0+10}" y="{y0+18}" font-size="12" fill="#ff7f0e">NPU</text>')
    body.append(f'<text x="{x0+60}" y="{y0+18}" font-size="12" fill="#1f77b4">GPU</text>')
    body.append(
        f'<text x="{x0}" y="{y0+h+48}" font-size="11" fill="#555">'
        "判定：|NPU线性度−GPU线性度|×100 ≤ 2 个百分点。第 1 行差 3.20pp → 不达标；第 2 行待 GPU。</text>"
    )
    body.append("</svg>")
    out.write_text("\n".join(body))


def main() -> None:
    specs = [
        ("row1_debugmodel", "第 1 行 muse_spark_debugmodel（4 卡）loss 对照",
         DATA / "gpu_debugmodel_1000.txt", DATA / "npu_debugmodel_1000.txt"),
        ("row2_1_with_cp", "第 2① muse_spark_debugmodel_with_cp（4 卡）loss 对照",
         DATA / "gpu_with_cp_1000.txt", DATA / "npu_with_cp_1000.txt"),
        ("row2_2_fsdp2", "第 2② muse_spark_debugmodel_fsdp2_cp2_ep4（4 卡）loss 对照",
         DATA / "gpu_fsdp2_cp2_ep4_1000.txt", DATA / "npu_fsdp2_cp2_ep4_1000.txt"),
        ("row2_3_mm_fsdp2", "第 2③ muse_spark_debugmodel_mm_fsdp2_tp2_cp2_ep4（8 卡）loss 对照",
         DATA / "gpu_mm_fsdp2_tp2_cp2_ep4_1000.txt", DATA / "npu_mm_fsdp2_tp2_cp2_ep4_1000.txt"),
    ]
    stats = {}
    for key, title, gpu, npu in specs:
        stats[key] = make_loss_panel(title, gpu, npu, FIGS / f"{key}_loss.svg")
        print(key, stats[key])
    make_perf_chart(FIGS / "row1_2_linearity.svg")
    print("wrote figs to", FIGS)


if __name__ == "__main__":
    main()
