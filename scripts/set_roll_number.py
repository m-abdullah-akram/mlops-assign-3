"""Replace placeholders everywhere:  python scripts/set_roll_number.py 22i-1234 GITHUBUSER DAGSHUBUSER

22l-6632 -> roll number (project/folder/repo names), mabdullah-akram, ma92365.
A lower-case, underscore version is used where Docker requires it (compose project name).
"""
import re
import sys
from pathlib import Path

if len(sys.argv) != 4:
    sys.exit(__doc__)
roll, gh, dh = sys.argv[1:]
root = Path(__file__).resolve().parent.parent
lower = re.sub(r"[^a-z0-9_]", "_", roll.lower())
skip = {".git", "__pycache__", "outputs", "mlruns", "data"}
for p in root.rglob("*"):
    if p.is_file() and not (set(p.relative_to(root).parts) & skip) and p.suffix in {
            ".py", ".md", ".yml", ".yaml", ".txt", ".example", ".cfg", ".sh", ".html", ""}:
        try:
            t = p.read_text()
        except UnicodeDecodeError:
            continue
        n = t.replace("22l_6632_a2_ops", f"{lower}_a2_ops").replace("22l-6632", roll)
        n = n.replace("mabdullah-akram", gh).replace("ma92365", dh)
        if n != t:
            p.write_text(n)
            print("updated", p.relative_to(root))
print("Now rename this folder to", f"{roll}_A2_Ops")
