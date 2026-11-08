import io
import unittest

from api.api_server import create_api
from tests.helpers import FakePredictor, make_image_bytes


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = create_api(predictor=FakePredictor(), api_key="s3cret").test_client()

    def call(self, key):
        headers = {"X-API-Key": key} if key is not None else {}
        return self.client.post("/predict", headers=headers, content_type="multipart/form-data",
                                data={"file": (io.BytesIO(make_image_bytes()), "x.png")})

    def test_health_is_public(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_requires_key(self):
        self.assertEqual(self.call(None).status_code, 401)
        self.assertEqual(self.call("wrong").status_code, 401)

    def test_predict_ok(self):
        r = self.call("s3cret")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["label"], "ai")

    def test_bad_file(self):
        r = self.client.post("/predict", headers={"X-API-Key": "s3cret"}, content_type="multipart/form-data",
                             data={"file": (io.BytesIO(b"junk"), "x.png")})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
