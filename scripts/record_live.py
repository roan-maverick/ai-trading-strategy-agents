"""Record the `atsa live` dashboard as an MP4 (Twitter/X) and GIF (README).

Requirements: ffmpeg, plus `pip install playwright && playwright install chromium`.

    # zero-setup demo recording (what docs/assets/atsa_live.* was made with)
    python scripts/record_live.py --demo

    # record a REAL hunt with your model and live crypto prices
    GEMINI_API_KEY=... python scripts/record_live.py --provider gemini --model gemini-4-neon \
        --source binance --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT --seconds 40 --speed 6
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from atsa.cli import _demo_settings
from atsa.config import load_settings
from atsa.llm import get_provider
from atsa.tui import run_live


async def svgs_to_pngs(svg_dir: Path, png_dir: Path, scale: float) -> None:
    from playwright.async_api import async_playwright

    png_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        browser = await p.chromium.launch(executable_path=chrome) if chrome else await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=scale)
        for f in sorted(svg_dir.glob("frame_*.svg")):
            await page.set_content(f"<html><body style='margin:0;background:#0d1117'>{f.read_text()}</body></html>")
            svg = await page.query_selector("svg")
            await svg.screenshot(path=str(png_dir / f.name.replace(".svg", ".png")))
        await browser.close()


def encode(work: Path, out: Path, speed: float, hold: float) -> None:
    ts = [float(x) for x in (work / "svg" / "timings.txt").read_text().split()]
    lines = []
    for i in range(1, len(ts)):
        d = hold if i == len(ts) - 1 else (ts[i + 1] - ts[i]) / speed
        lines.append(f"file '{work / 'png' / f'frame_{i:05d}.png'}'\nduration {d:.4f}")
    lines.append(f"file '{work / 'png' / f'frame_{len(ts) - 1:05d}.png'}'")
    (work / "list.txt").write_text("\n".join(lines) + "\n")
    base = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt")]
    subprocess.run(base + ["-vf", "fps=30,scale=1920:-2:flags=lanczos,format=yuv420p", "-c:v", "libx264",
                           "-preset", "veryfast", "-crf", "18", "-movflags", "+faststart", str(out.with_suffix(".mp4"))],
                   check=True)
    subprocess.run(base + ["-vf", "fps=12,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:"
                                  "stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                           "-loop", "0", str(out.with_suffix(".gif"))], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline agents + synthetic market (no API key)")
    ap.add_argument("--provider"), ap.add_argument("--model"), ap.add_argument("--source")
    ap.add_argument("--symbols"), ap.add_argument("--interval")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=None, help="stop recording after N seconds")
    ap.add_argument("--speed", type=float, default=3.0, help="playback speed-up")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds to hold the final frame")
    ap.add_argument("--out", default="docs/assets/atsa_live")
    args = ap.parse_args()

    if args.demo:
        s = _demo_settings(Path(".atsa-demo"), args.generations, 7,
                           ["DEMO-A", "DEMO-B", "DEMO-C", "DEMO-D", "DEMO-E", "DEMO-F"])
    else:
        s = load_settings()
        for key, section, attr in (("provider", "llm", "provider"), ("model", "llm", "model"),
                                   ("source", "data", "source"), ("interval", "data", "interval")):
            if getattr(args, key):
                setattr(getattr(s, section), attr, getattr(args, key))
        if args.symbols:
            s.data.symbols = args.symbols.split(",")
        s.swarm.generations = args.generations
    s.runtime.parallel_symbols = args.parallel
    provider = get_provider(s.llm, seed=s.swarm.seed)
    if args.demo:
        provider.latency = (0.25, 0.9)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        asyncio.run(run_live(s, provider, fps=12, record_dir=str(work / "svg"), record_seconds=args.seconds,
                             size=(150, 36), linger=2.0))
        asyncio.run(svgs_to_pngs(work / "svg", work / "png", scale=1.5))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        encode(work, out, args.speed, args.hold)
    print(f"wrote {out.with_suffix('.mp4')} and {out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
