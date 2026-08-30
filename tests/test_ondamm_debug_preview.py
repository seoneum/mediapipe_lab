from __future__ import annotations

import json
import sys
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_debug_preview import DebugPreviewServer  # noqa: E402


class OndammDebugPreviewTests(unittest.TestCase):
    def test_debug_preview_does_not_open_second_camera(self) -> None:
        with patch("cv2.VideoCapture") as video_capture:
            server = DebugPreviewServer(port=0)
            server.start()
            try:
                server.publish(np.zeros((24, 32, 3), dtype=np.uint8))
            finally:
                server.stop()
        video_capture.assert_not_called()

    def test_non_loopback_bind_requires_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow_remote_bind"):
            DebugPreviewServer(host="0.0.0.0", port=8766)

    def test_local_http_serves_health_snapshot_and_mjpeg(self) -> None:
        server = DebugPreviewServer(port=0)
        server.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            server.publish(np.full((24, 32, 3), 90, dtype=np.uint8))
            self.assertTrue(server.wait_for_frame())

            connection.request("GET", "/health")
            response = connection.getresponse()
            health = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertTrue(health["frame_available"])
            self.assertEqual(health["camera_owner"], "external-runtime")
            self.assertFalse(health["disk_persistence"])

            connection.request("GET", "/snapshot.jpg")
            response = connection.getresponse()
            snapshot = response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "image/jpeg")
            self.assertTrue(snapshot.startswith(b"\xff\xd8"))

            connection.request("GET", "/preview.mjpg")
            response = connection.getresponse()
            first_bytes = response.read(80)
            self.assertEqual(response.status, 200)
            self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
            self.assertTrue(first_bytes.startswith(b"--frame\r\n"))
        finally:
            connection.close()
            server.stop()

    def test_debug_preview_is_throttled(self) -> None:
        server = DebugPreviewServer(port=0, max_fps=10.0)
        server.start()
        try:
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            self.assertTrue(server.publish(frame))
            self.assertFalse(server.publish(frame))
            self.assertEqual(server.accepted_frame_count, 1)
            self.assertTrue(server.wait_for_frame())
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
