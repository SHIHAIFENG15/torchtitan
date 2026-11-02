#!/usr/bin/env python3
"""Generate SVG charts for Muse Spark acceptance report (rows 1-5)."""
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
    diff = [a - b for a, b in zip(nl, gl)]
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
        axes(x1, y1, w, h, 1, n, dmin, dmax, "step", "NPU - GPU loss"),
        polyline(gs, diff, x1, y1, w, h, 1, n, dmin, dmax, "#d62728", 1.4),
        f'<line x1="{x1}" y1="{y1 + h - (0 - dmin)/(dmax-dmin)*h:.1f}" '
        f'x2="{x1+w}" y2="{y1 + h - (0 - dmin)/(dmax-dmin)*h:.1f}" '
        f'stroke="#999" stroke-dasharray="4 3"/>',
        f'<text x="{x1+10}" y="{y1+18}" font-size="12">'
        f"MAE={mae:.5f}  MRE={mre*100:.2f}%</text>",
        "</svg>",
    ]
    out.write_text("\n".join(body))
    return {"mae": mae, "mre": mre, "n": n}


def make_perf_chart(out: Path) -> None:
    # name, n4, n8, g4, g8, delta_pp, verdict  (None = N/A)
    rows = [
        ("1 debugmodel", 322.89, 307.98, 1003.27, 924.81, 3.20, "fail"),
        ("2(1) with_cp", 148.53, 144.21, 485.99, 478.41, 1.35, "pass"),
        ("2(2) fsdp2", 121.64, 116.24, 335.02, 324.08, 1.17, "pass"),
        ("2(3) mm_fsdp2", 47.94, 38.86, 134.18, 109.40, 0.47, "pass"),
        ("3 Dist-MoE", None, None, None, None, None, "na"),
        ("4 debugmodel_mm", 118.56, 114.96, 374.36, 358.08, 1.31, "pass"),
        ("5 ultradata", 378.25, 341.40, 10230.50, 9874.95, 6.26, "fail"),
    ]
    W, H = 1180, 480
    x0, y0, w, h = 70, 55, 1040, 300
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        f'<text x="{W/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold">'
        "8/4 card linearity (stable tps, step&gt;100)</text>",
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#ccc"/>',
    ]
    ymin, ymax = 0.75, 1.02
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        py = y0 + h - (yv - ymin) / (ymax - ymin) * h
        body.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+w}" y2="{py:.1f}" stroke="#eee"/>')
        body.append(f'<text x="{x0-8}" y="{py+4:.1f}" text-anchor="end" font-size="10">{yv:.2f}</text>')
    n_groups = len(rows)
    group_w = w / n_groups
    for gi, (name, n4, n8, g4, g8, delta, verdict) in enumerate(rows):
        cx = x0 + group_w * (gi + 0.5)
        bw = group_w * 0.26
        if verdict == "na" or n4 is None:
            body.append(
                f'<text x="{cx:.1f}" y="{y0+h/2:.1f}" text-anchor="middle" font-size="11" fill="#888">'
                "N/A</text>"
            )
        else:
            n_ratio = n8 / n4
            g_ratio = g8 / g4
            nh = (n_ratio - ymin) / (ymax - ymin) * h
            gh = (g_ratio - ymin) / (ymax - ymin) * h
            body.append(
                f'<rect x="{cx-bw-3:.1f}" y="{y0+h-nh:.1f}" width="{bw:.1f}" height="{nh:.1f}" '
                f'fill="#ff7f0e"/>'
            )
            body.append(
                f'<text x="{cx-bw/2-3:.1f}" y="{y0+h-nh-6:.1f}" text-anchor="middle" font-size="9">'
                f"{n_ratio:.4f}</text>"
            )
            body.append(
                f'<rect x="{cx+3:.1f}" y="{y0+h-gh:.1f}" width="{bw:.1f}" height="{gh:.1f}" '
                f'fill="#1f77b4"/>'
            )
            body.append(
                f'<text x="{cx+3+bw/2:.1f}" y="{y0+h-gh-6:.1f}" text-anchor="middle" font-size="9">'
                f"{g_ratio:.4f}</text>"
            )
            color = "#c62828" if verdict == "fail" else "#2e7d32"
            body.append(
                f'<text x="{cx:.1f}" y="{y0+h+36}" text-anchor="middle" font-size="10" fill="{color}">'
                f"delta {delta:.2f}pp</text>"
            )
        body.append(
            f'<text x="{cx:.1f}" y="{y0+h+20}" text-anchor="middle" font-size="10">'
            f"{svg_escape(name)}</text>"
        )
    body.append(f'<text x="{x0+10}" y="{y0+18}" font-size="12" fill="#ff7f0e">NPU</text>')
    body.append(f'<text x="{x0+60}" y="{y0+18}" font-size="12" fill="#1f77b4">GPU</text>')
    body.append(
        f'<text x="{x0}" y="{y0+h+58}" font-size="11" fill="#555">'
        "Pass if |NPU-GPU|x100 &lt;= 2pp. Fail: row1 3.20pp, row5 6.26pp. Row3 Dist-MoE no GPU baseline.</text>"
    )
    body.append("</svg>")
    out.write_text("\n".join(body))


def make_memory_chart(out: Path) -> None:
    # end/stable single-card reserved GiB
    rows = [
        ("1", 0.26, 0.24, "pass"),
        ("2(1)", 0.25, 3.85, "failA"),
        ("2(2)", 0.25, 3.85, "failA"),
        ("2(3)", 0.28, 14.15, "failA"),
        ("3", 0.09, None, "na"),
        ("4", 0.28, 14.15, "failA"),
        ("5", 13.25, 0.54, "failB"),
    ]
    W, H = 1180, 460
    x0, y0, w, h = 70, 50, 1040, 300
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        f'<text x="{W/2}" y="28" text-anchor="middle" font-size="16" font-weight="bold">'
        "Single-card log memory reserved (GiB)</text>",
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#ccc"/>',
    ]
    ymin, ymax = 0.0, 16.0
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        py = y0 + h - (yv - ymin) / (ymax - ymin) * h
        body.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+w}" y2="{py:.1f}" stroke="#eee"/>')
        body.append(f'<text x="{x0-8}" y="{py+4:.1f}" text-anchor="end" font-size="10">{yv:.0f}</text>')
    n_groups = len(rows)
    group_w = w / n_groups
    for gi, (name, npu, gpu, verdict) in enumerate(rows):
        cx = x0 + group_w * (gi + 0.5)
        bw = group_w * 0.26
        nh = (npu - ymin) / (ymax - ymin) * h
        body.append(
            f'<rect x="{cx-bw-3:.1f}" y="{y0+h-nh:.1f}" width="{bw:.1f}" height="{nh:.1f}" '
            f'fill="#ff7f0e"/>'
        )
        body.append(
            f'<text x="{cx-bw/2-3:.1f}" y="{y0+h-nh-6:.1f}" text-anchor="middle" font-size="9">'
            f"{npu:.2f}</text>"
        )
        if gpu is not None:
            gh = (gpu - ymin) / (ymax - ymin) * h
            body.append(
                f'<rect x="{cx+3:.1f}" y="{y0+h-gh:.1f}" width="{bw:.1f}" height="{gh:.1f}" '
                f'fill="#1f77b4"/>'
            )
            body.append(
                f'<text x="{cx+3+bw/2:.1f}" y="{y0+h-gh-6:.1f}" text-anchor="middle" font-size="9">'
                f"{gpu:.2f}</text>"
            )
        else:
            body.append(
                f'<text x="{cx+bw/2:.1f}" y="{y0+h-12:.1f}" text-anchor="middle" font-size="10" fill="#888">'
                "N/A</text>"
            )
        label = {"pass": "pass", "failA": "fail A", "failB": "fail B", "na": "N/A"}[verdict]
        color = {"pass": "#2e7d32", "failA": "#c62828", "failB": "#c62828", "na": "#888"}[verdict]
        body.append(
            f'<text x="{cx:.1f}" y="{y0+h+20}" text-anchor="middle" font-size="10">{svg_escape(name)}</text>'
        )
        body.append(
            f'<text x="{cx:.1f}" y="{y0+h+36}" text-anchor="middle" font-size="10" fill="{color}">'
            f"{label}</text>"
        )
    body.append(f'<text x="{x0+10}" y="{y0+18}" font-size="12" fill="#ff7f0e">NPU</text>')
    body.append(f'<text x="{x0+60}" y="{y0+18}" font-size="12" fill="#1f77b4">GPU</text>')
    body.append(
        f'<text x="{x0}" y="{y0+h+58}" font-size="11" fill="#555">'
        "A: GPU climb (CP/flex). B: NPU much higher (32K LoRA). Row3: NPU-only 0.09GiB, no GPU.</text>"
    )
    body.append("</svg>")
    out.write_text("\n".join(body))


def main() -> None:
    specs = [
        ("row1_debugmodel", "Row1 debugmodel loss",
         DATA / "gpu_debugmodel_1000.txt", DATA / "npu_debugmodel_1000.txt"),
        ("row2_1_with_cp", "Row2(1) with_cp loss",
         DATA / "gpu_with_cp_1000.txt", DATA / "npu_with_cp_1000.txt"),
        ("row2_2_fsdp2", "Row2(2) fsdp2_cp2_ep4 loss",
         DATA / "gpu_fsdp2_cp2_ep4_1000.txt", DATA / "npu_fsdp2_cp2_ep4_1000.txt"),
        ("row2_3_mm_fsdp2", "Row2(3) mm_fsdp2 loss",
         DATA / "gpu_mm_fsdp2_tp2_cp2_ep4_1000.txt", DATA / "npu_mm_fsdp2_tp2_cp2_ep4_1000.txt"),
        ("row4_debugmodel_mm", "Row4 debugmodel_mm loss",
         DATA / "gpu_debugmodel_mm_1000.txt", DATA / "npu_debugmodel_mm_1000.txt"),
        ("row5_ultradata", "Row5 ultradata_lora_sft loss",
         DATA / "gpu_ultradata_lora_1000.txt", DATA / "npu_ultradata_lora_1000.txt"),
    ]
    for key, title, gpu, npu in specs:
        stats = make_loss_panel(title, gpu, npu, FIGS / f"{key}_loss.svg")
        print(key, stats)
    make_perf_chart(FIGS / "rows_linearity.svg")
    make_memory_chart(FIGS / "rows_memory.svg")
    # keep old names as aliases for links that may still point there
    (FIGS / "row1_2_linearity.svg").write_text((FIGS / "rows_linearity.svg").read_text())
    (FIGS / "row1_2_memory.svg").write_text((FIGS / "rows_memory.svg").read_text())
    print("wrote figs to", FIGS)


if __name__ == "__main__":
    main()
