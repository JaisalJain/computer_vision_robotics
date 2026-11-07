#!/usr/bin/env python
"""Tidy executed notebooks before committing them (outputs are kept).

* removes ``stderr`` stream outputs (library warnings, progress bars),
* collapses carriage-return progress lines in ``stdout`` to their final state,
* drops interactive-widget outputs and per-cell execution timing metadata,
* downscales embedded PNG images wider than ``--max-width`` pixels.

Usage::

    python scripts/nb_clean.py projects/*/notebooks/*.ipynb [--keep-stderr] [--max-width 1400]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path


def collapse_cr(text: str) -> str:
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))


def shrink_png(b64: str, max_width: int) -> str:
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    if img.width <= max_width:
        return b64
    h = round(img.height * max_width / img.width)
    img = img.resize((max_width, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def clean(path: Path, keep_stderr: bool, max_width: int) -> tuple[int, int]:
    before = path.stat().st_size
    nb = json.loads(path.read_text())
    for cell in nb.get("cells", []):
        cell.get("metadata", {}).pop("execution", None)
        if cell.get("cell_type") != "code":
            continue
        outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                if out.get("name") == "stderr" and not keep_stderr:
                    continue
                text = out["text"]
                text = collapse_cr("".join(text) if isinstance(text, list) else text)
                if not text.strip():
                    continue
                out["text"] = text.splitlines(keepends=True)
            data = out.get("data")
            if data is not None:
                for key in [k for k in data if k.startswith("application/vnd.jupyter.widget")]:
                    del data[key]
                if "image/png" in data and max_width:
                    png = data["image/png"]
                    png = "".join(png) if isinstance(png, list) else png
                    data["image/png"] = shrink_png(png, max_width)
                if not data:
                    continue
            outputs.append(out)
        cell["outputs"] = outputs
    nb.get("metadata", {}).pop("widgets", None)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return before, path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--keep-stderr", action="store_true")
    ap.add_argument("--max-width", type=int, default=1400)
    args = ap.parse_args()
    for nb in args.notebooks:
        b, a = clean(nb, args.keep_stderr, args.max_width)
        print(f"{nb}: {b / 1024:.0f} KiB -> {a / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
