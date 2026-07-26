from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_web import ApiRouter, OndammWebService, make_http_handler  # noqa: E402


class OndammHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_context = tempfile.TemporaryDirectory(prefix="ondamm-http-test-")
        self.ui_dir = Path(self.temp_context.name)
        (self.ui_dir / "index.html").write_text("<!doctype html><title>ON DAMM test</title>", encoding="utf-8")
        self.media_path = self.ui_dir / "event.mp4"
        self.media_path.write_bytes(b"0123456789")
        service = OndammWebService()
        service.resolve_media_clip = lambda clip_id: self.media_path if clip_id == "clip-test" else None
        handler = make_http_handler(ApiRouter(service), self.ui_dir)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp_context.cleanup()

    def test_serves_health_json(self) -> None:
        self.connection.request("GET", "/api/health")
        response = self.connection.getresponse()
        payload = json.loads(response.read())

        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
        self.assertEqual(payload["status"], "ok")

    def test_serves_single_page_app_index(self) -> None:
        self.connection.request("GET", "/")
        response = self.connection.getresponse()
        html = response.read().decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("ON DAMM test", html)

    def test_streams_local_clip_with_http_range(self) -> None:
        self.connection.request("GET", "/media/clips/clip-test", headers={"Range": "bytes=2-5"})
        response = self.connection.getresponse()
        content = response.read()

        self.assertEqual(response.status, 206)
        self.assertEqual(response.getheader("Content-Type"), "video/mp4")
        self.assertEqual(response.getheader("Content-Range"), "bytes 2-5/10")
        self.assertEqual(content, b"2345")


if __name__ == "__main__":
    unittest.main()
