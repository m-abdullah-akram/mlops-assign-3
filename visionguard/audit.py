"""Task 1.2 - dataset audit: one row per image with size, shape, mode, format, SHA-256."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from visionguard import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def class_of(parts):
    """Map the folders above a file to a class name ('ai' / 'real'); fall back to the parent."""
    for part in parts:
        if part.lower() in config.FOLDER_TO_CLASS:
            return config.FOLDER_TO_CLASS[part.lower()]
    return parts[-1].lower() if parts else "unknown"


def audit(data_dir):
    data_dir = Path(data_dir)
    rows = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(data_dir)
        row = {
            "path": p.as_posix(), "class_name": class_of(rel.parts[:-1]),
            "file_size": p.stat().st_size, "sha256": sha256_file(p),
            "width": None, "height": None, "mode": None, "format": None,
            "corrupt": False, "error": "",
        }
        try:
            with Image.open(p) as im:
                im.verify()          # cheap structural check
            with Image.open(p) as im:
                im.load()            # full decode catches truncated files
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
        except Exception as exc:  # noqa: BLE001 - any failure means "unreadable"
            row.update(corrupt=True, error=repr(exc)[:200])
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["is_duplicate"] = df.duplicated("sha256", keep="first")
    classes_per_hash = df.groupby("sha256")["class_name"].transform("nunique")
    df["label_conflict"] = classes_per_hash > 1   # same bytes, different class folder
    return df


def summarize(df):
    out = []
    out.append(f"Total files: {len(df)}")
    out.append("\nImages per class:\n" + df["class_name"].value_counts().to_string())
    ok = df[~df["corrupt"]]
    stats = ok.groupby("class_name")[["width", "height", "file_size"]].agg(["min", "median", "max"])
    out.append("\nWidth / height / file size (bytes) per class:\n" + stats.to_string())
    out.append(f"\nCorrupt / unreadable files: {int(df['corrupt'].sum())}")
    out.append(f"Exact duplicates (extra copies): {int(df['is_duplicate'].sum())}")
    out.append(f"Files whose bytes appear in more than one class: {int(df['label_conflict'].sum())}")
    out.append("\nColour modes:\n" + ok["mode"].value_counts().to_string())
    out.append("\nFormats:\n" + ok["format"].value_counts().to_string())
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "audit.csv"))
    args = ap.parse_args()
    df = audit(args.data_dir)
    if df.empty:
        raise SystemExit(f"No images found under {args.data_dir}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = summarize(df)
    Path(args.out).with_name("audit_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
