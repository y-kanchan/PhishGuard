"""Zip extension/ for the Chrome Web Store (or for sharing).

    python scripts/package_extension.py
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"
REQUIRED = ["manifest.json", "model/phishguard.onnx", "model/meta.json", "lib/ort.wasm.min.js",
            "lib/ort-wasm-simd-threaded.wasm", "lib/ort-wasm-simd-threaded.mjs", "icons/icon128.png"]


def main():
    missing = [f for f in REQUIRED if not (EXT / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing} - run scripts/export_onnx.py / scripts/make_icons.py first (see README)")
    version = json.loads((EXT / "manifest.json").read_text())["version"]
    out = ROOT / "dist" / f"phishguard-extension-{version}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(EXT.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(EXT).as_posix())
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
