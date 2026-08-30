"""Local-only HTTP preview for an annotated frame owned by the camera runtime.

This module never opens a camera and never writes frames to disk.  The live
runtime publishes only its already-rendered preview frame; a browser connected
through an SSH tunnel can read the latest JPEG or an MJPEG stream.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class _PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True


class _LatestJpeg:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.data: bytes | None = None
        self.sequence = 0
        self.stopping = False

    def publish(self, data: bytes) -> None:
        with self.condition:
            self.data = data
            self.sequence += 1
            self.condition.notify_all()

    def stop(self) -> None:
        with self.condition:
            self.stopping = True
            self.condition.notify_all()

    def wait_after(self, sequence: int, *, timeout: float = 1.0) -> tuple[int, bytes | None, bool]:
        with self.condition:
            if self.sequence <= sequence and not self.stopping:
                self.condition.wait(timeout=timeout)
            return self.sequence, self.data, self.stopping


def _handler_for(state: _LatestJpeg) -> type[BaseHTTPRequestHandler]:
    class DebugPreviewHandler(BaseHTTPRequestHandler):
        server_version = "ONDAMM-DebugPreview/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._serve_index()
            elif path == "/health":
                self._serve_health()
            elif path == "/snapshot.jpg":
                self._serve_snapshot()
            elif path == "/preview.mjpg":
                self._serve_mjpeg()
            else:
                self.send_error(404)

        def _serve_index(self) -> None:
            content = (
                "<!doctype html><html lang='ko'><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>ON DAMM 실시간 관찰 화면</title>"
                "<style>body{margin:0;background:#07111b;color:#eaf7f3;font-family:sans-serif}"
                "main{max-width:1200px;margin:auto;padding:20px}img{width:100%;height:auto;"
                "background:#000;border:1px solid #4ee2be}p{color:#a9c4bd}</style>"
                "<main><h1>ON DAMM 실시간 미세 움직임 관찰</h1>"
                "<p>디버그·발표용 로컬 화면입니다. 전체 세션 영상은 저장하지 않습니다.</p>"
                "<img src='/preview.mjpg' alt='실시간 얼굴 landmark와 움직임 상태'></main></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _serve_health(self) -> None:
            with state.condition:
                payload: dict[str, Any] = {
                    "status": "ok",
                    "service": "ondamm-debug-preview",
                    "frame_available": state.data is not None,
                    "sequence": state.sequence,
                    "disk_persistence": False,
                    "camera_owner": "external-runtime",
                }
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _serve_snapshot(self) -> None:
            with state.condition:
                data = state.data
            if data is None:
                self.send_error(503, "preview frame is not available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _serve_mjpeg(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            sequence = -1
            try:
                while True:
                    sequence, data, stopping = state.wait_after(sequence)
                    if stopping:
                        return
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def log_message(self, format: str, *args: object) -> None:
            return

    return DebugPreviewHandler


class DebugPreviewServer:
    """Publish annotated frames over an opt-in local HTTP server."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8766,
        jpeg_quality: int = 82,
        max_fps: float = 10.0,
        allow_remote_bind: bool = False,
    ) -> None:
        host = str(host).strip()
        if not host:
            raise ValueError("debug preview host is required")
        if host not in LOCAL_HOSTS and not allow_remote_bind:
            raise ValueError(
                "non-loopback debug preview bind requires explicit allow_remote_bind=True"
            )
        if not 0 <= int(port) <= 65535:
            raise ValueError("debug preview port must be between 0 and 65535")
        if not 30 <= int(jpeg_quality) <= 95:
            raise ValueError("debug preview JPEG quality must be between 30 and 95")
        if max_fps <= 0:
            raise ValueError("debug preview max_fps must be positive")
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.max_fps = float(max_fps)
        self._state = _LatestJpeg()
        self._server: _PreviewHttpServer | None = None
        self._thread: threading.Thread | None = None
        self._encoder_thread: threading.Thread | None = None
        self._pending_condition = threading.Condition()
        self._pending_frame: np.ndarray | None = None
        self._encoder_stopping = False
        self._last_accepted_at = float("-inf")
        self._accepted_count = 0

    @property
    def server_port(self) -> int:
        return int(self._server.server_port) if self._server is not None else self.port

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host == "localhost" else self.host
        if ":" in display_host and not display_host.startswith("["):
            display_host = f"[{display_host}]"
        return f"http://{display_host}:{self.server_port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _PreviewHttpServer((self.host, self.port), _handler_for(self._state))
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ondamm-debug-preview",
            daemon=True,
        )
        self._thread.start()
        self._encoder_stopping = False
        self._encoder_thread = threading.Thread(
            target=self._encode_loop,
            name="ondamm-debug-preview-jpeg",
            daemon=True,
        )
        self._encoder_thread.start()

    @property
    def accepted_frame_count(self) -> int:
        return self._accepted_count

    def wait_for_frame(self, *, timeout: float = 2.0) -> bool:
        """Wait for the asynchronous encoder in tests/startup health checks."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._state.condition:
            while self._state.data is None and not self._state.stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._state.condition.wait(timeout=remaining)
            return self._state.data is not None

    def publish(self, frame: np.ndarray) -> bool:
        """Queue only the latest due frame; JPEG work stays off the camera loop."""
        values = np.asarray(frame)
        if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
            raise ValueError("debug preview frame must be a uint8 BGR image")
        now = time.monotonic()
        with self._pending_condition:
            if now - self._last_accepted_at + 1e-9 < 1.0 / self.max_fps:
                return False
            self._last_accepted_at = now
            self._accepted_count += 1
            self._pending_frame = np.array(values, copy=True)
            self._pending_condition.notify()
        return True

    def _encode_loop(self) -> None:
        while True:
            with self._pending_condition:
                while self._pending_frame is None and not self._encoder_stopping:
                    self._pending_condition.wait(timeout=1.0)
                if self._encoder_stopping:
                    return
                frame = self._pending_frame
                self._pending_frame = None
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if ok:
                self._state.publish(encoded.tobytes())

    def stop(self) -> None:
        with self._pending_condition:
            self._encoder_stopping = True
            self._pending_frame = None
            self._pending_condition.notify_all()
        self._state.stop()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=3)
        self._server = None
        self._thread = None
        self._encoder_thread = None

    def __enter__(self) -> "DebugPreviewServer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.stop()
        return False
