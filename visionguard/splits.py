"""Build leak-free train/val/test lists (CSV) from the audit table.

Rules: drop corrupt files, drop exact duplicates (keep the first copy), drop images whose bytes
appear under both classes, drop formats TensorFlow cannot decode, then split 70/15/15 stratified.
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from visionguard import config


def build_splits(df, seed=config.SEED, val_frac=0.15, test_frac=0.15):
    keep = df[~df["corrupt"] & ~df["is_duplicate"] & ~df["label_conflict"]
              & df["format"].isin(config.TF_FORMATS)].copy()
    keep["label"] = keep["class_name"].map({n: i for i, n in enumerate(config.CLASS_NAMES)})
    keep = keep.dropna(subset=["label"])
    keep["label"] = keep["label"].astype(int)
    keep["path"] = [os.path.relpath(p, config.ROOT).replace("\\", "/") for p in keep["path"]]
    cols = ["path", "label", "class_name", "sha256"]
    train_val, test = train_test_split(keep[cols], test_size=test_frac, stratify=keep["label"],
                                       random_state=seed)
    train, val = train_test_split(train_val, test_size=val_frac / (1 - test_frac),
                                  stratify=train_val["label"], random_state=seed)
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", default=str(config.REPORTS_DIR / "audit.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.audit)
    train, val, test = build_splits(df)
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
        print(f"{name:5s} {len(part):5d} images  {part['class_name'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
