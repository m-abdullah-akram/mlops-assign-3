import io
import tempfile
import unittest
from pathlib import Path

from app.app import create_app
from tests.helpers import FakePredictor, make_image_bytes


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app({"DATABASE_URL": f"sqlite:///{Path(self.tmp.name) / 't.db'}",
                               "MAX_UPLOAD_MB": 0.01, "TESTING": True}, predictor=FakePredictor())
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def post(self, data, name):
        return self.client.post("/predict", data={"file": (io.BytesIO(data), name)},
                                content_type="multipart/form-data")

    def test_pages_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/history").status_code, 200)
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["model_loaded"])
        self.assertEqual(self.client.get("/nope").status_code, 404)

    def test_valid_upload_is_logged_and_shown_in_history(self):
        r = self.post(make_image_bytes("PNG"), "../../evil name.png")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Likely AI-generated", r.data)
        self.assertEqual(self.app.extensions["db"].count(), 1)
        hist = self.client.get("/history").data
        self.assertIn(b"evil_name.png", hist)          # sanitized file name
        self.assertNotIn(b"..", hist.split(b"<tbody>")[1])
        self.assertIn(b"fake_model v7", hist)

    def test_wrong_extension_rejected(self):
        r = self.post(b"hello", "notes.txt")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.app.extensions["db"].count(), 0)

    def test_fake_image_that_cannot_be_decoded_rejected(self):
        r = self.post(b"this is not an image", "photo.jpg")
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"cannot be decoded", r.data)

    def test_truncated_image_rejected(self):
        r = self.post(make_image_bytes("PNG", (300, 300))[:120], "cut.png")
        self.assertEqual(r.status_code, 400)

    def test_too_large_rejected(self):
        r = self.post(b"\xff" * 50_000, "big.jpg")
        self.assertEqual(r.status_code, 413)
        self.assertIn(b"File too large", r.data)

    def test_missing_file(self):
        r = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_nothing_written_to_disk(self):
        before = set(Path(self.tmp.name).iterdir())
        self.post(make_image_bytes("JPEG"), "a.jpg")
        self.assertEqual(set(Path(self.tmp.name).iterdir()), before)   # only the sqlite file exists


if __name__ == "__main__":
    unittest.main()
