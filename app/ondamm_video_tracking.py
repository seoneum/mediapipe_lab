"""ON DAMM video pipeline — person tracking with persistent global IDs.

Wraps YOLO26 detection + BoT-SORT (static-camera config) for short-term track
association, and re-associates track identities across detector ID switches /
occlusions with L2-normalized ArcFace (insightface buffalo_l) embedding
centroids (``GlobalIdResolver``). Emits the frozen ``TrackObservation`` schema
consumed by downstream todos (facial signals, metric fusion, renderer).

Identity policy: body/appearance ReID inside the tracker is NOT used as
identity (``with_reid: False`` in configs/botsort_static.yaml); identity comes
only from face-embedding centroids here. Offline file processing only — no
network calls at runtime beyond first-run local weight loads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_WEIGHTS = "models/yolo26s.pt"
DEFAULT_TRACKER_CONFIG = "configs/botsort_static.yaml"
PERSON_CLASS_ID = 0


class VideoInputError(Exception):
    """Raised when a video input cannot be opened or decoded."""


@dataclass(frozen=True)
class TrackObservation:
    """Frozen cross-module schema (see plan `## Execution strategy`)."""

    frame_idx: int
    ts_sec: float
    track_id: int
    global_id: str
    bbox_xyxy: list[float]  # [x1, y1, x2, y2]
    det_conf: float
    low_confidence: bool


# ---------------------------------------------------------------------------
# Video input
# ---------------------------------------------------------------------------


class FrameReader:
    """Iterate ``(frame_idx, ts_sec, frame)`` tuples from a video file.

    ``fps`` is captured from the container (fallback 30.0 when the container
    reports 0). Raises :class:`VideoInputError` at construction when the video
    cannot be opened OR the first frame cannot be decoded (truncated/corrupt
    files that merely open are rejected too).
    """

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = str(video_path)
        self.capture = cv2.VideoCapture(self.video_path)
        try:
            if not self.capture.isOpened():
                raise VideoInputError(f"cannot open video: {self.video_path}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if fps and fps > 0 else 30.0
            ok, first = self.capture.read()
            if not ok or first is None:
                raise VideoInputError(f"cannot open video: {self.video_path}")
        except Exception:
            self.capture.release()
            raise
        self._pending_first = first
        self._next_idx = 0

    def __iter__(self) -> "FrameReader":
        return self

    def __next__(self) -> tuple[int, float, np.ndarray]:
        if self._pending_first is not None:
            frame, self._pending_first = self._pending_first, None
        else:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.close()
                raise StopIteration
        idx = self._next_idx
        self._next_idx += 1
        return idx, idx / self.fps, frame

    def close(self) -> None:
        self._pending_first = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def padded_crop(
    frame: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    pad_ratio: float = 0.10,
) -> np.ndarray | None:
    """Return the bbox crop expanded by ``pad_ratio`` on each side, clamped."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    px, py = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    xa = max(0, int(math.floor(x1 - px)))
    ya = max(0, int(math.floor(y1 - py)))
    xb = min(w, int(math.ceil(x2 + px)))
    yb = min(h, int(math.ceil(y2 + py)))
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb].copy()


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("embedding has zero/non-finite norm")
    return v / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


# ---------------------------------------------------------------------------
# Face embedding (ArcFace via insightface buffalo_l, det+rec only)
# ---------------------------------------------------------------------------


class FaceEmbedder:
    """512-d L2-normalized face embedding from a bbox crop (pad 10%, clamp).

    Uses insightface ``FaceAnalysis("buffalo_l")`` restricted to detection +
    recognition modules. Weights load locally from ``~/.insightface/models``
    on first use (no runtime network beyond that first-run load).
    """

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        pad_ratio: float = 0.10,
        ctx_id: int = -1,
    ) -> None:
        from insightface.app import FaceAnalysis  # deferred: heavy import

        self.pad_ratio = pad_ratio
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def embed_crop(self, frame_bgr: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray | None:
        """Return the L2-normalized 512-d embedding for the best face in the
        padded crop, or ``None`` when no face is found."""
        crop = padded_crop(frame_bgr, bbox_xyxy, self.pad_ratio)
        if crop is None:
            return None
        faces = self._app.get(crop)
        if not faces:
            return None
        best = max(faces, key=lambda f: float(f.det_score))
        return l2_normalize(np.asarray(best.normed_embedding, dtype=np.float32))


# ---------------------------------------------------------------------------
# Persistent global identity resolution
# ---------------------------------------------------------------------------


class GlobalIdResolver:
    """Bind short-lived tracker IDs to persistent global IDs via face centroids.

    - Each track keeps an EMA centroid of its embeddings (alpha=0.9).
    - Each global identity keeps an EMA centroid updated by its member tracks.
    - A new/unknown track binds to an existing global ID iff the cosine between
      its EMA centroid and the best global centroid is >= ``bind_threshold``
      AND best minus second_best >= ``margin`` (a single candidate auto-passes
      the margin gate). Otherwise the track keeps/gets a provisional
      ``unknown_<n>`` id flagged ``low_confidence=True``.
    - Late re-binding is attempted on every subsequent embedded observation of
      a provisionally-bound track, so identity can recover at any later frame.
    - ``low_confidence`` is True only while a track's assignment is a
      provisional mint; a confident bind to an existing cluster clears it.
    """

    EMA_ALPHA = 0.9

    def __init__(self, bind_threshold: float = 0.45, margin: float = 0.05) -> None:
        self.bind_threshold = float(bind_threshold)
        self.margin = float(margin)
        self._track_centroid: dict[int, np.ndarray] = {}
        self._global_centroid: dict[str, np.ndarray] = {}
        self._track_global: dict[int, str] = {}
        self._track_low: dict[int, bool] = {}
        self._unknown_count = 0

    # -- public API ---------------------------------------------------------

    def update(self, track_id: int, embedding: np.ndarray | None) -> tuple[str, bool]:
        """Feed one observation for ``track_id``; return ``(global_id, low_confidence)``.

        With ``embedding=None`` the existing assignment is carried forward (or a
        provisional id is minted so the schema always carries a concrete id).
        """
        current = self._track_global.get(track_id)
        if embedding is None:
            if current is not None:
                return current, self._track_low.get(track_id, True)
            return self._mint_unknown(track_id), True

        emb = l2_normalize(embedding)
        prev_track = self._track_centroid.get(track_id)
        if prev_track is None:
            track_cent = emb
        else:
            track_cent = l2_normalize(self.EMA_ALPHA * prev_track + (1.0 - self.EMA_ALPHA) * emb)
        self._track_centroid[track_id] = track_cent

        target, bound = self._resolve_target(track_id, track_cent)
        if target is None:
            target = current if current is not None else self._mint_unknown(track_id)
        self._track_global[track_id] = target

        prev_low = self._track_low.get(track_id)
        low = False if bound else (prev_low if prev_low is not None else True)
        self._track_low[track_id] = low

        prev_glob = self._global_centroid.get(target)
        if prev_glob is None:
            self._global_centroid[target] = track_cent.copy()
        else:
            self._global_centroid[target] = l2_normalize(
                self.EMA_ALPHA * prev_glob + (1.0 - self.EMA_ALPHA) * emb
            )
        return target, low

    def current(self, track_id: int) -> tuple[str, bool]:
        """Assignment without a new embedding (carry forward / provision)."""
        return self.update(track_id, None)

    def centroid_of(self, global_id: str) -> np.ndarray | None:
        return self._global_centroid.get(global_id)

    def is_provisional(self, global_id: str) -> bool:
        return global_id.startswith("unknown_")

    # -- internals ----------------------------------------------------------

    def _resolve_target(self, track_id: int, track_cent: np.ndarray) -> tuple[str | None, bool]:
        """Return ``(target_global_id_or_None, confidently_bound)``."""
        current = self._track_global.get(track_id)
        if not self._global_centroid:
            return None, False

        gids = list(self._global_centroid)
        sims = [cosine(track_cent, self._global_centroid[g]) for g in gids]
        order = sorted(range(len(gids)), key=lambda i: sims[i], reverse=True)
        best_gid, best_sim = gids[order[0]], sims[order[0]]
        second_sim = sims[order[1]] if len(order) > 1 else None
        margin_ok = second_sim is None or (best_sim - second_sim) >= self.margin

        if best_sim >= self.bind_threshold and margin_ok:
            return best_gid, True  # bind or late re-bind (sticky clusters win naturally)
        return None, False  # no confident match: caller keeps prior or mints

    def _mint_unknown(self, track_id: int) -> str:
        gid = f"unknown_{self._unknown_count}"
        self._unknown_count += 1
        self._track_global[track_id] = gid
        return gid


# ---------------------------------------------------------------------------
# Detector + tracker wiring
# ---------------------------------------------------------------------------


def _scalar(value: object) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _quad(value: object) -> list[float]:
    seq = value.tolist() if hasattr(value, "tolist") else value
    return [_scalar(v) for v in seq]


class PersonTracker:
    """YOLO26 + BoT-SORT(static) tracking emitting :class:`TrackObservation`.

    ``model`` may be injected for hermetic tests (must expose
    ``track(frame, **kwargs) -> list`` of results with ``.boxes`` carrying
    ``id``/``conf``/``xyxy``). When omitted, ultralytics YOLO loads
    ``weights`` lazily on first construction.
    """

    def __init__(
        self,
        device: str | None = None,
        weights: str = DEFAULT_WEIGHTS,
        tracker_config: str = DEFAULT_TRACKER_CONFIG,
        embedder: FaceEmbedder | None = None,
        resolver: GlobalIdResolver | None = None,
        sample_every: int = 15,
        fps: float = 30.0,
        model: object | None = None,
    ) -> None:
        self.device = device
        self.weights = weights
        self.tracker_config = tracker_config
        self.embedder = embedder
        self.resolver = resolver if resolver is not None else GlobalIdResolver()
        self.sample_every = int(sample_every)
        self.fps = float(fps)
        self._model = model
        self._last_embed_frame: dict[int, int] = {}

    def _ensure_model(self) -> object:
        if self._model is None:
            from ultralytics import YOLO  # deferred: heavy import

            self._model = YOLO(self.weights)
        return self._model

    def _detect(self, frame: np.ndarray) -> list[tuple[int, list[float], float]]:
        model = self._ensure_model()
        kwargs: dict[str, object] = {
            "persist": True,
            "classes": [PERSON_CLASS_ID],
            "tracker": self.tracker_config,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = model.track(frame, **kwargs)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        ids = boxes.id
        dets: list[tuple[int, list[float], float]] = []
        synthetic_id = -1
        for i in range(len(boxes)):
            if ids is not None:
                track_id = int(_scalar(ids[i]))
            else:
                track_id = synthetic_id
                synthetic_id -= 1
            dets.append((track_id, _quad(boxes.xyxy[i]), _scalar(boxes.conf[i])))
        return dets

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        ts_sec: float | None = None,
    ) -> list[TrackObservation]:
        """Run detection+tracking (+identity resolution) on one BGR frame."""
        if ts_sec is None:
            ts_sec = frame_idx / self.fps
        observations: list[TrackObservation] = []
        for track_id, bbox, conf in self._detect(frame):
            embedding = self._maybe_embed(track_id, frame, bbox, frame_idx)
            global_id, low_confidence = self.resolver.update(track_id, embedding)
            observations.append(
                TrackObservation(
                    frame_idx=int(frame_idx),
                    ts_sec=float(ts_sec),
                    track_id=int(track_id),
                    global_id=global_id,
                    bbox_xyxy=bbox,
                    det_conf=float(conf),
                    low_confidence=bool(low_confidence),
                )
            )
        return observations

    def _maybe_embed(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: list[float],
        frame_idx: int,
    ) -> np.ndarray | None:
        if self.embedder is None:
            return None
        last = self._last_embed_frame.get(track_id)
        if last is not None and (frame_idx - last) < self.sample_every:
            return None
        embedding = self.embedder.embed_crop(frame, bbox)
        if embedding is not None:
            self._last_embed_frame[track_id] = frame_idx
        return embedding
