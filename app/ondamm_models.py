from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


@dataclass
class SessionSummary:
    session_id: str
    title: str
    activity_name: str
    observed_response: str
    educator_interpretation: str
    approved_by: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        title: str,
        activity_name: str,
        observed_response: str,
        educator_interpretation: str,
        approved_by: str,
        tags: list[str] | None = None,
    ) -> "SessionSummary":
        return cls(
            session_id=f"session-{uuid4().hex[:8]}",
            title=title.strip(),
            activity_name=activity_name.strip(),
            observed_response=observed_response.strip(),
            educator_interpretation=educator_interpretation.strip(),
            approved_by=approved_by.strip(),
            tags=unique_preserving_order(tags or []),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSummary":
        return cls(
            session_id=data["session_id"],
            title=data["title"],
            activity_name=data["activity_name"],
            observed_response=data["observed_response"],
            educator_interpretation=data["educator_interpretation"],
            approved_by=data["approved_by"],
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", utc_now()),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = unique_preserving_order(self.tags)
        return payload


@dataclass
class RecommendationEntry:
    recommendation_id: str
    goal: str
    summary: str
    suggested_activities: list[str]
    rationale_lines: list[str]
    drafted_by: str
    approved_by: str | None = None
    status: str = "draft"
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        goal: str,
        summary: str,
        suggested_activities: list[str],
        rationale_lines: list[str],
        drafted_by: str,
        approved_by: str | None = None,
    ) -> "RecommendationEntry":
        return cls(
            recommendation_id=f"plan-{uuid4().hex[:8]}",
            goal=goal.strip(),
            summary=summary.strip(),
            suggested_activities=unique_preserving_order(suggested_activities),
            rationale_lines=unique_preserving_order(rationale_lines),
            drafted_by=drafted_by.strip(),
            approved_by=approved_by.strip() if approved_by else None,
            status="approved" if approved_by else "draft",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecommendationEntry":
        return cls(
            recommendation_id=data["recommendation_id"],
            goal=data["goal"],
            summary=data["summary"],
            suggested_activities=list(data.get("suggested_activities", [])),
            rationale_lines=list(data.get("rationale_lines", [])),
            drafted_by=data["drafted_by"],
            approved_by=data.get("approved_by"),
            status=data.get("status", "draft"),
            created_at=data.get("created_at", utc_now()),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suggested_activities"] = unique_preserving_order(self.suggested_activities)
        payload["rationale_lines"] = unique_preserving_order(self.rationale_lines)
        return payload


@dataclass
class Dossier:
    child_id: str
    display_name: str
    age_band: str
    communication_modality: str
    local_canonical_id: str = field(default_factory=lambda: f"local-{uuid4().hex[:10]}")
    canonical_status: str = "active"
    confirmed_preferences: list[str] = field(default_factory=list)
    confirmed_avoidances: list[str] = field(default_factory=list)
    effective_strategies: list[str] = field(default_factory=list)
    triggers_and_calming_supports: list[str] = field(default_factory=list)
    handoff_notes: list[str] = field(default_factory=list)
    approved_session_summaries: list[SessionSummary] = field(default_factory=list)
    approved_plan_history: list[RecommendationEntry] = field(default_factory=list)
    access_audit_records: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        child_id: str,
        display_name: str,
        age_band: str,
        communication_modality: str,
        confirmed_preferences: list[str] | None = None,
        confirmed_avoidances: list[str] | None = None,
        effective_strategies: list[str] | None = None,
        triggers_and_calming_supports: list[str] | None = None,
        handoff_notes: list[str] | None = None,
    ) -> "Dossier":
        return cls(
            child_id=child_id.strip(),
            display_name=display_name.strip(),
            age_band=age_band.strip(),
            communication_modality=communication_modality.strip(),
            confirmed_preferences=unique_preserving_order(confirmed_preferences or []),
            confirmed_avoidances=unique_preserving_order(confirmed_avoidances or []),
            effective_strategies=unique_preserving_order(effective_strategies or []),
            triggers_and_calming_supports=unique_preserving_order(triggers_and_calming_supports or []),
            handoff_notes=unique_preserving_order(handoff_notes or []),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dossier":
        return cls(
            child_id=data["child_id"],
            display_name=data["display_name"],
            age_band=data["age_band"],
            communication_modality=data["communication_modality"],
            local_canonical_id=data.get("local_canonical_id", f"local-{uuid4().hex[:10]}"),
            canonical_status=data.get("canonical_status", "active"),
            confirmed_preferences=list(data.get("confirmed_preferences", [])),
            confirmed_avoidances=list(data.get("confirmed_avoidances", [])),
            effective_strategies=list(data.get("effective_strategies", [])),
            triggers_and_calming_supports=list(data.get("triggers_and_calming_supports", [])),
            handoff_notes=list(data.get("handoff_notes", [])),
            approved_session_summaries=[
                SessionSummary.from_dict(item)
                for item in data.get("approved_session_summaries", [])
            ],
            approved_plan_history=[
                RecommendationEntry.from_dict(item)
                for item in data.get("approved_plan_history", [])
            ],
            access_audit_records=list(data.get("access_audit_records", [])),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )

    def touch(self) -> None:
        self.updated_at = utc_now()

    def add_session_summary(self, summary: SessionSummary) -> None:
        self.approved_session_summaries.append(summary)
        self.touch()

    def add_recommendation(self, recommendation: RecommendationEntry) -> None:
        self.approved_plan_history.append(recommendation)
        self.touch()

    def add_audit_event(self, event_type: str, actor_id: str, details: dict[str, Any]) -> None:
        self.access_audit_records.append(
            {
                "event_type": event_type,
                "actor_id": actor_id,
                "timestamp": utc_now(),
                "details": details,
            }
        )
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "child_id": self.child_id,
            "display_name": self.display_name,
            "age_band": self.age_band,
            "communication_modality": self.communication_modality,
            "local_canonical_id": self.local_canonical_id,
            "canonical_status": self.canonical_status,
            "confirmed_preferences": unique_preserving_order(self.confirmed_preferences),
            "confirmed_avoidances": unique_preserving_order(self.confirmed_avoidances),
            "effective_strategies": unique_preserving_order(self.effective_strategies),
            "triggers_and_calming_supports": unique_preserving_order(self.triggers_and_calming_supports),
            "handoff_notes": unique_preserving_order(self.handoff_notes),
            "approved_session_summaries": [item.to_dict() for item in self.approved_session_summaries],
            "approved_plan_history": [item.to_dict() for item in self.approved_plan_history],
            "access_audit_records": self.access_audit_records,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
