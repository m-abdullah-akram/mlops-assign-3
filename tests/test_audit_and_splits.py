import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_image_bytes
from visionguard import audit
from visionguard.splits import build_splits


class AuditTests(unittest.TestCase):
    def test_audit_finds_corrupt_and_duplicates_and_builds_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AiArtData").mkdir()
            (root / "RealArt").mkdir()
            for i in range(20):
                (root / "AiArtData" / f"a{i}.png").write_bytes(make_image_bytes("PNG", (32 + i, 32), (i * 9, 0, 0)))
                (root / "RealArt" / f"r{i}.jpg").write_bytes(make_image_bytes("JPEG", (40 + i, 30), (0, i * 9, 0)))
            (root / "RealArt" / "dup.jpg").write_bytes((root / "RealArt" / "r0.jpg").read_bytes())
            (root / "AiArtData" / "broken.png").write_bytes(b"not a png")
            df = audit.audit(root)
            self.assertEqual(len(df), 42)
            self.assertEqual(int(df["corrupt"].sum()), 1)
            self.assertEqual(int(df["is_duplicate"].sum()), 1)
            self.assertEqual(set(df["class_name"]), {"ai", "real"})
            self.assertIn("Corrupt", audit.summarize(df))
            # splits: no corrupt, no duplicate, no overlap between partitions
            train, val, test = build_splits(df)
            total = len(train) + len(val) + len(test)
            self.assertEqual(total, 40)
            hashes = [set(x["sha256"]) for x in (train, val, test)]
            self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
            self.assertTrue(set(train["label"]) == {0, 1})


if __name__ == "__main__":
    unittest.main()
