"""Per-person temporal movement prototypes and incremental unknown clusters.

This store is separate from FacialMovementProfile and from support-strategy
learning.  Vectors stay in a local model store; public JSON exposes only digests,
counts, distances, timestamps, and human-review lifecycle state.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from ondamm_movement_explanation import compare_region_profiles


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str, name: str) -> str:
    cleaned = str(value).strip()
    if not SAFE_ID.fullmatch(cleaned):
        raise ValueError(f"{name} contains unsupported characters")
    return cleaned


def _vector(values: Sequence[float], *, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("embedding must be a finite 1D vector")
    if dimension is not None and result.size != dimension:
        raise ValueError(f"embedding dimension must be {dimension}")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-8:
        raise ValueError("embedding must not be zero")
    return result / norm


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(1.0 - float(np.dot(left, right)), 0.0, 2.0))


def _digest_vector(vector: np.ndarray) -> str:
    canonical = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class PatternMemoryPolicy:
    known_distance_threshold: float = 0.25
    candidate_distance_threshold: float = 0.05
    suppression_distance_threshold: float = 0.15
    min_occurrences_for_clip: int = 3
    strong_candidate_occurrences: int = 5
    max_candidates: int = 128

    def __post_init__(self) -> None:
        for name in (
            "known_distance_threshold",
            "candidate_distance_threshold",
            "suppression_distance_threshold",
        ):
            value = getattr(self, name)
            if not 0 < value <= 2:
                raise ValueError(f"{name} must be in (0, 2]")
        if self.min_occurrences_for_clip < 2:
            raise ValueError("min_occurrences_for_clip must be at least two")
        if self.strong_candidate_occurrences < self.min_occurrences_for_clip:
            raise ValueError("strong_candidate_occurrences must not be smaller than clip threshold")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")


@dataclass(frozen=True)
class PatternDecision:
    lifecycle: str
    episode_id: str
    pattern_id: str | None
    candidate_id: str | None
    occurrence_count: int
    distance: float | None
    novelty_score: float
    clip_required: bool
    strong_candidate: bool
    nearest_known_pattern: str | None = None
    nearest_known_distance: float | None = None
    movement_summary: dict[str, Any] | None = None
    regional_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycle": self.lifecycle,
            "episode_id": self.episode_id,
            "pattern_id": self.pattern_id,
            "candidate_id": self.candidate_id,
            "occurrence_count": self.occurrence_count,
            "distance": self.distance,
            "novelty_score": self.novelty_score,
            "clip_required": self.clip_required,
            "strong_candidate": self.strong_candidate,
            "nearest_known_pattern": self.nearest_known_pattern,
            "nearest_known_distance": self.nearest_known_distance,
            "movement_summary": self.movement_summary,
            "regional_comparison": self.regional_comparison,
        }


class PatternMemoryStore:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: Path,
        *,
        child_id: str,
        encoder_digest: str,
        embedding_dimension: int,
        policy: PatternMemoryPolicy | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.child_id = _safe_id(child_id, "child_id")
        if not re.fullmatch(r"[0-9a-f]{16,128}", encoder_digest):
            raise ValueError("encoder_digest must be hexadecimal")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        self.encoder_digest = encoder_digest
        self.embedding_dimension = int(embedding_dimension)
        self.policy = policy or PatternMemoryPolicy()
        self.directory = self.root / self.child_id
        self.metadata_path = self.directory / "memory.json"
        self.vectors_path = self.directory / "vectors.npz"
        self._lock = threading.RLock()

    @classmethod
    def open_existing(cls, root: Path, *, child_id: str) -> "PatternMemoryStore":
        directory = root.expanduser().resolve() / _safe_id(child_id, "child_id")
        path = directory / "memory.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing temporal pattern memory for {child_id}")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read pattern memory metadata") from exc
        raw_policy = metadata.get("policy", {})
        return cls(
            root,
            child_id=child_id,
            encoder_digest=str(metadata.get("encoder_digest", "")),
            embedding_dimension=int(metadata.get("embedding_dimension", 0)),
            policy=PatternMemoryPolicy(
                known_distance_threshold=float(raw_policy.get("known_distance_threshold", 0.25)),
                candidate_distance_threshold=float(raw_policy.get("candidate_distance_threshold", 0.05)),
                suppression_distance_threshold=float(raw_policy.get("suppression_distance_threshold", 0.15)),
                min_occurrences_for_clip=int(raw_policy.get("min_occurrences_for_clip", 3)),
                strong_candidate_occurrences=int(raw_policy.get("strong_candidate_occurrences", 5)),
                max_candidates=int(raw_policy.get("max_candidates", 128)),
            ),
        )

    def observe_episode(
        self,
        *,
        episode_id: str,
        embedding: Sequence[float],
        start_timestamp: float,
        end_timestamp: float,
        quality_score: float,
        movement_summary: Mapping[str, Any] | None = None,
    ) -> PatternDecision:
        episode_id = _safe_id(episode_id, "episode_id")
        vector = _vector(embedding, dimension=self.embedding_dimension)
        if not all(math.isfinite(float(value)) for value in (start_timestamp, end_timestamp, quality_score)):
            raise ValueError("episode metadata must be finite")
        if end_timestamp < start_timestamp:
            raise ValueError("episode end must not precede start")
        with self._lock:
            metadata, vectors = self._load()
            previous = metadata["seen_episodes"].get(episode_id)
            if previous:
                return PatternDecision(**previous)

            known = self._nearest(metadata["known_patterns"], vectors, vector)
            if known and known[1] <= float(known[0].get("distance_threshold", self.policy.known_distance_threshold)):
                item, distance = known
                regional_comparison = compare_region_profiles(
                    movement_summary,
                    item.get("movement_profile"),
                )
                item["occurrence_count"] = int(item.get("occurrence_count", 0)) + 1
                item["last_seen"] = utc_now()
                self._update_movement_profile(item, movement_summary)
                decision = PatternDecision(
                    lifecycle="KNOWN_OCCURRENCE",
                    episode_id=episode_id,
                    pattern_id=item["pattern_id"],
                    candidate_id=None,
                    occurrence_count=item["occurrence_count"],
                    distance=round(distance, 6),
                    novelty_score=round(distance, 6),
                    clip_required=False,
                    strong_candidate=False,
                    nearest_known_pattern=item["pattern_id"],
                    nearest_known_distance=round(distance, 6),
                    movement_summary=dict(movement_summary) if movement_summary else None,
                    regional_comparison=regional_comparison,
                )
                return self._remember_decision(metadata, vectors, decision)

            suppressed = self._nearest(metadata["suppressed"], vectors, vector)
            if suppressed and suppressed[1] <= self.policy.suppression_distance_threshold:
                item, distance = suppressed
                decision = PatternDecision(
                    lifecycle="SUPPRESSED",
                    episode_id=episode_id,
                    pattern_id=None,
                    candidate_id=item["suppression_id"],
                    occurrence_count=int(item.get("match_count", 0)) + 1,
                    distance=round(distance, 6),
                    novelty_score=round(distance, 6),
                    clip_required=False,
                    strong_candidate=False,
                    nearest_known_pattern=known[0]["pattern_id"] if known else None,
                    nearest_known_distance=round(known[1], 6) if known else None,
                )
                item["match_count"] = decision.occurrence_count
                item["last_seen"] = utc_now()
                return self._remember_decision(metadata, vectors, decision)

            nearest_known_distance = known[1] if known else 1.0
            candidate = self._nearest(metadata["candidates"], vectors, vector)
            if candidate and candidate[1] <= self.policy.candidate_distance_threshold:
                item, candidate_distance = candidate
                count = int(item["occurrence_count"])
                regional_comparison = compare_region_profiles(
                    movement_summary,
                    item.get("movement_profile"),
                )
                old_centroid = vectors[item["vector_key"]]
                # Preserve the resultant length so repeated normalization does
                # not turn the update into an unweighted blend of the previous
                # direction and only the newest exemplar.
                resultant_norm = float(item.get("resultant_norm", count))
                raw_sum = old_centroid * resultant_norm + vector
                updated_resultant_norm = float(np.linalg.norm(raw_sum))
                updated = _vector(raw_sum)
                vectors[item["vector_key"]] = updated
                item["occurrence_count"] = count + 1
                item["resultant_norm"] = round(updated_resultant_norm, 6)
                item["last_seen"] = utc_now()
                item["mean_duration_seconds"] = round(
                    (float(item["mean_duration_seconds"]) * count + (end_timestamp - start_timestamp)) / (count + 1),
                    6,
                )
                item["mean_quality_score"] = round(
                    (float(item["mean_quality_score"]) * count + quality_score) / (count + 1),
                    6,
                )
                item["prototype_digest"] = _digest_vector(updated)
                sample_count = int(item.get("distance_sample_count", 0)) + 1
                previous_mean = float(item.get("distance_mean", 0.0))
                delta = float(candidate_distance) - previous_mean
                updated_mean = previous_mean + delta / sample_count
                item["distance_sample_count"] = sample_count
                item["distance_mean"] = round(updated_mean, 6)
                item["distance_m2"] = round(
                    float(item.get("distance_m2", 0.0)) + delta * (float(candidate_distance) - updated_mean),
                    8,
                )
                item["recommended_distance_threshold"] = self._recommended_threshold(item)
                self._update_movement_profile(item, movement_summary)
                candidate_id = item["candidate_id"]
            else:
                if len(metadata["candidates"]) >= self.policy.max_candidates:
                    raise RuntimeError("candidate memory capacity reached; review or suppress existing candidates")
                candidate_id = f"candidate-{uuid4().hex[:10]}"
                vector_key = f"candidate__{candidate_id}"
                vectors[vector_key] = vector
                candidate_distance = None
                regional_comparison = None
                item = {
                    "candidate_id": candidate_id,
                    "vector_key": vector_key,
                    "prototype_digest": _digest_vector(vector),
                    "occurrence_count": 1,
                    "resultant_norm": 1.0,
                    "first_seen": utc_now(),
                    "last_seen": utc_now(),
                    "mean_duration_seconds": round(end_timestamp - start_timestamp, 6),
                    "mean_quality_score": round(float(np.clip(quality_score, 0.0, 1.0)), 6),
                    "nearest_known_pattern": known[0]["pattern_id"] if known else None,
                    "nearest_known_distance": round(nearest_known_distance, 6),
                    "review_state": "observing",
                    "source_event_ids": [],
                    "distance_sample_count": 0,
                    "distance_mean": 0.0,
                    "distance_m2": 0.0,
                    "recommended_distance_threshold": round(
                        min(0.05, self.policy.known_distance_threshold),
                        6,
                    ),
                    "movement_profile": self._movement_distribution(movement_summary),
                    "movement_profile_count": 1 if self._movement_distribution(movement_summary) else 0,
                }
                metadata["candidates"].append(item)

            count = int(item["occurrence_count"])
            clip_required = (
                count >= self.policy.min_occurrences_for_clip
                and not item.get("source_event_ids")
            )
            lifecycle = "REPEATING_CANDIDATE" if count >= self.policy.min_occurrences_for_clip else "UNKNOWN_OCCURRENCE"
            decision = PatternDecision(
                lifecycle=lifecycle,
                episode_id=episode_id,
                pattern_id=None,
                candidate_id=candidate_id,
                occurrence_count=count,
                distance=round(float(candidate_distance), 6) if candidate_distance is not None else None,
                novelty_score=round(float(nearest_known_distance), 6),
                clip_required=clip_required,
                strong_candidate=count >= self.policy.strong_candidate_occurrences,
                nearest_known_pattern=known[0]["pattern_id"] if known else None,
                nearest_known_distance=round(float(nearest_known_distance), 6),
                movement_summary=dict(movement_summary) if movement_summary else None,
                regional_comparison=regional_comparison,
            )
            return self._remember_decision(metadata, vectors, decision)

    def attach_source_event(self, *, candidate_id: str, event_id: str) -> dict[str, Any]:
        with self._lock:
            metadata, vectors = self._load()
            candidate = self._candidate(metadata, candidate_id)
            event_id = _safe_id(event_id, "event_id")
            if event_id not in candidate["source_event_ids"]:
                candidate["source_event_ids"].append(event_id)
            candidate["review_state"] = "review_pending"
            self._save(metadata, vectors)
            return self._public_candidate(candidate)

    def mark_watch(self, *, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            metadata, vectors = self._load()
            candidate = self._candidate(metadata, candidate_id)
            candidate["review_state"] = "watch"
            self._save(metadata, vectors)
            return self._public_candidate(candidate)

    def suppress_candidate(self, *, candidate_id: str, approved_by: str, reason: str) -> dict[str, Any]:
        with self._lock:
            metadata, vectors = self._load()
            candidate = self._candidate(metadata, candidate_id)
            approved_by = str(approved_by).strip()
            reason = str(reason).strip()
            if not approved_by or not reason:
                raise ValueError("approved_by and reason are required")
            vector = vectors[candidate["vector_key"]]
            suppression_id = f"suppression-{uuid4().hex[:10]}"
            vector_key = f"suppressed__{suppression_id}"
            vectors[vector_key] = vector
            record = {
                "suppression_id": suppression_id,
                "vector_key": vector_key,
                "prototype_digest": _digest_vector(vector),
                "source_candidate_id": candidate_id,
                "approved_by": approved_by[:120],
                "reason": reason[:500],
                "created_at": utc_now(),
                "last_seen": None,
                "match_count": 0,
            }
            metadata["suppressed"].append(record)
            self._remove_candidate(metadata, vectors, candidate)
            self._save(metadata, vectors)
            return {key: value for key, value in record.items() if key != "vector_key"}

    def promote_candidate(
        self,
        *,
        candidate_id: str,
        display_name: str,
        approved_by: str,
        source_event_ids: Sequence[str],
        distance_threshold: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            metadata, vectors = self._load()
            candidate = self._candidate(metadata, candidate_id)
            display_name = str(display_name).strip()
            approved_by = str(approved_by).strip()
            if not display_name or len(display_name) > 120:
                raise ValueError("display_name is required and must be at most 120 characters")
            if not approved_by or len(approved_by) > 120:
                raise ValueError("approved_by is required and must be at most 120 characters")
            if int(candidate["occurrence_count"]) < self.policy.min_occurrences_for_clip:
                raise RuntimeError("candidate has not reached the recurrence threshold")
            canonical_events = [_safe_id(value, "source_event_id") for value in source_event_ids]
            if not canonical_events or not set(canonical_events).issubset(set(candidate["source_event_ids"])):
                raise ValueError("source_event_ids must reference recorded events for this candidate")
            threshold = self._recommended_threshold(candidate) if distance_threshold is None else float(distance_threshold)
            if not 0 < threshold <= 2:
                raise ValueError("distance_threshold must be in (0, 2]")
            pattern_id = f"pattern-{uuid4().hex[:10]}"
            vector = vectors[candidate["vector_key"]]
            vector_key = f"known__{pattern_id}"
            vectors[vector_key] = vector
            record = {
                "pattern_id": pattern_id,
                "display_name": display_name,
                "encoder_digest": self.encoder_digest,
                "vector_key": vector_key,
                "prototype_digest": _digest_vector(vector),
                "distance_threshold": round(threshold, 6),
                "support_count": int(candidate["occurrence_count"]),
                "occurrence_count": 0,
                "source_event_ids": list(dict.fromkeys(canonical_events)),
                "approved_by": approved_by,
                "created_at": utc_now(),
                "last_seen": None,
                "movement_profile": candidate.get("movement_profile"),
                "movement_profile_count": int(candidate.get("movement_profile_count", 0)),
            }
            metadata["known_patterns"].append(record)
            self._remove_candidate(metadata, vectors, candidate)
            self._save(metadata, vectors)
            return self._public_known(record)

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            metadata, _ = self._load()
            return {
                "schema_version": self.SCHEMA_VERSION,
                "child_id": self.child_id,
                "encoder_digest": self.encoder_digest,
                "embedding_dimension": self.embedding_dimension,
                "policy": {
                    "known_distance_threshold": self.policy.known_distance_threshold,
                    "candidate_distance_threshold": self.policy.candidate_distance_threshold,
                    "suppression_distance_threshold": self.policy.suppression_distance_threshold,
                    "min_occurrences_for_clip": self.policy.min_occurrences_for_clip,
                    "strong_candidate_occurrences": self.policy.strong_candidate_occurrences,
                    "max_candidates": self.policy.max_candidates,
                },
                "known_patterns": [self._public_known(item) for item in metadata["known_patterns"]],
                "candidates": [self._public_candidate(item) for item in metadata["candidates"]],
                "suppressed": [
                    {key: value for key, value in item.items() if key != "vector_key"}
                    for item in metadata["suppressed"]
                ],
                "raw_media_saved_for_unpromoted_counts": False,
                "online_tcn_retraining": False,
            }

    def _remember_decision(
        self,
        metadata: dict[str, Any],
        vectors: dict[str, np.ndarray],
        decision: PatternDecision,
    ) -> PatternDecision:
        metadata["seen_episodes"][decision.episode_id] = decision.to_dict()
        # Bound idempotency history without affecting occurrence counts.
        while len(metadata["seen_episodes"]) > 4096:
            metadata["seen_episodes"].pop(next(iter(metadata["seen_episodes"])))
        self._save(metadata, vectors)
        return decision

    def _load(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if not self.metadata_path.is_file():
            return self._empty_metadata(), {}
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("could not read pattern memory metadata") from exc
        if metadata.get("encoder_digest") != self.encoder_digest:
            raise RuntimeError("pattern memory encoder digest does not match the active encoder")
        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            raise RuntimeError("unsupported temporal pattern memory schema")
        if metadata.get("child_id") != self.child_id:
            raise RuntimeError("pattern memory child_id does not match the active child")
        if int(metadata.get("embedding_dimension", 0)) != self.embedding_dimension:
            raise RuntimeError("pattern memory embedding dimension does not match the active encoder")
        vectors: dict[str, np.ndarray] = {}
        if self.vectors_path.is_file():
            with np.load(self.vectors_path, allow_pickle=False) as archive:
                vectors = {key: _vector(archive[key], dimension=self.embedding_dimension) for key in archive.files}
        for collection in ("known_patterns", "candidates", "suppressed"):
            for item in metadata.get(collection, []):
                if item.get("vector_key") not in vectors:
                    raise RuntimeError(f"pattern memory vector missing for {collection}")
        return metadata, vectors

    def _save(self, metadata: dict[str, Any], vectors: dict[str, np.ndarray]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        vector_temp = self.directory / ".vectors.tmp.npz"
        np.savez_compressed(vector_temp, **vectors)
        vector_temp.replace(self.vectors_path)
        metadata["updated_at"] = utc_now()
        metadata_temp = self.directory / ".memory.tmp"
        metadata_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata_temp.replace(self.metadata_path)

    def _empty_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "child_id": self.child_id,
            "encoder_digest": self.encoder_digest,
            "embedding_dimension": self.embedding_dimension,
            "policy": {
                "known_distance_threshold": self.policy.known_distance_threshold,
                "candidate_distance_threshold": self.policy.candidate_distance_threshold,
                "suppression_distance_threshold": self.policy.suppression_distance_threshold,
                "min_occurrences_for_clip": self.policy.min_occurrences_for_clip,
                "strong_candidate_occurrences": self.policy.strong_candidate_occurrences,
                "max_candidates": self.policy.max_candidates,
            },
            "known_patterns": [],
            "candidates": [],
            "suppressed": [],
            "seen_episodes": {},
            "created_at": utc_now(),
            "updated_at": None,
        }

    @staticmethod
    def _nearest(
        items: list[dict[str, Any]],
        vectors: dict[str, np.ndarray],
        vector: np.ndarray,
    ) -> tuple[dict[str, Any], float] | None:
        if not items:
            return None
        ranked = [(item, cosine_distance(vector, vectors[item["vector_key"]])) for item in items]
        return min(ranked, key=lambda pair: pair[1])

    def _recommended_threshold(self, candidate: dict[str, Any]) -> float:
        """Child/candidate-specific known threshold from exemplar spread."""
        sample_count = int(candidate.get("distance_sample_count", 0))
        mean = max(0.0, float(candidate.get("distance_mean", 0.0)))
        variance = (
            max(0.0, float(candidate.get("distance_m2", 0.0))) / (sample_count - 1)
            if sample_count > 1
            else 0.0
        )
        # The policy value is a safety ceiling, not the value applied to every
        # child.  The small floor supports near-identical approved exemplars.
        learned = mean + 2.0 * math.sqrt(variance) + 0.02
        return round(float(np.clip(learned, 0.05, self.policy.known_distance_threshold)), 6)

    @staticmethod
    def _movement_distribution(movement_summary: Mapping[str, Any] | None) -> dict[str, float] | None:
        if not movement_summary:
            return None
        raw = movement_summary.get("region_distribution")
        if not isinstance(raw, Mapping):
            return None
        distribution: dict[str, float] = {}
        for key, value in raw.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number >= 0:
                distribution[str(key)] = number
        total = sum(distribution.values())
        if total <= 1e-12:
            return None
        return {key: round(value / total, 6) for key, value in distribution.items()}

    @classmethod
    def _update_movement_profile(
        cls,
        item: dict[str, Any],
        movement_summary: Mapping[str, Any] | None,
    ) -> None:
        current = cls._movement_distribution(movement_summary)
        if not current:
            return
        previous = item.get("movement_profile")
        count = int(item.get("movement_profile_count", 0))
        if not isinstance(previous, Mapping) or count <= 0:
            item["movement_profile"] = current
            item["movement_profile_count"] = 1
            return
        keys = set(previous) | set(current)
        updated = {
            key: (float(previous.get(key, 0.0)) * count + float(current.get(key, 0.0))) / (count + 1)
            for key in keys
        }
        total = sum(updated.values())
        item["movement_profile"] = {
            key: round(value / total, 6) for key, value in updated.items()
        }
        item["movement_profile_count"] = count + 1

    @staticmethod
    def _candidate(metadata: dict[str, Any], candidate_id: str) -> dict[str, Any]:
        candidate_id = _safe_id(candidate_id, "candidate_id")
        for item in metadata["candidates"]:
            if item["candidate_id"] == candidate_id:
                return item
        raise FileNotFoundError(f"Missing temporal movement candidate: {candidate_id}")

    @staticmethod
    def _remove_candidate(
        metadata: dict[str, Any],
        vectors: dict[str, np.ndarray],
        candidate: dict[str, Any],
    ) -> None:
        metadata["candidates"] = [
            item for item in metadata["candidates"] if item["candidate_id"] != candidate["candidate_id"]
        ]
        vectors.pop(candidate["vector_key"], None)

    @staticmethod
    def _public_known(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key != "vector_key"}

    @staticmethod
    def _public_candidate(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key != "vector_key"}
