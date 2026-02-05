"""Starter scaffolding for an enterprise AI decision & execution layer.

This is intentionally small but structured to map to the founder plan:
- Ingestion + normalization
- Knowledge graph + metrics
- Reasoning layer
- Policy / approvals
- Action execution

Replace in-memory stubs with real connectors (CRM/ERP/etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Signal:
    """A normalized event or metric from enterprise systems."""

    source: str
    timestamp: datetime
    payload: Dict[str, Any]


@dataclass
class Decision:
    """A decision recommendation produced by the reasoning layer."""

    summary: str
    rationale: str
    confidence: float
    recommended_actions: List[str]


@dataclass
class Approval:
    """Human-in-the-loop approval record."""

    decision_id: str
    approved_by: str
    approved_at: datetime
    notes: Optional[str] = None


@dataclass
class ActionResult:
    action: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


class IngestionConnector:
    """Base class for ingesting data from enterprise systems."""

    def fetch_signals(self) -> Iterable[Signal]:
        raise NotImplementedError


class InMemoryConnector(IngestionConnector):
    """Example connector for local testing or demos."""

    def __init__(self, signals: Iterable[Signal]):
        self._signals = list(signals)

    def fetch_signals(self) -> Iterable[Signal]:
        return self._signals


class KnowledgeGraph:
    """Minimal graph-like store; replace with real KG or graph DB."""

    def __init__(self) -> None:
        self._signals: List[Signal] = []

    def ingest(self, signals: Iterable[Signal]) -> None:
        self._signals.extend(signals)

    def recent_signals(self, limit: int = 50) -> List[Signal]:
        return sorted(self._signals, key=lambda s: s.timestamp, reverse=True)[:limit]


class ReasoningEngine:
    """Produces decisions based on signals + policies + heuristics."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self._kg = knowledge_graph

    def daily_brief(self) -> Decision:
        recent = self._kg.recent_signals(limit=5)
        if not recent:
            return Decision(
                summary="No critical risks detected today.",
                rationale="No recent signals were ingested.",
                confidence=0.2,
                recommended_actions=[],
            )

        top_signal = recent[0]
        return Decision(
            summary=f"Top risk: {top_signal.payload.get('title', 'Unknown risk')}",
            rationale=f"Latest signal from {top_signal.source} at {top_signal.timestamp:%Y-%m-%d %H:%M}.",
            confidence=0.72,
            recommended_actions=["notify_owner", "open_escalation"],
        )


class PolicyEngine:
    """Enforces approvals, guardrails, and auditability."""

    def requires_approval(self, action: str) -> bool:
        return action in {"open_escalation", "refund_customer", "update_contract"}

    def validate_action(self, action: str) -> bool:
        return action in {"notify_owner", "open_escalation", "create_task"}


class ActionExecutor:
    """Executes actions against external systems (stubbed)."""

    def execute(self, action: str) -> ActionResult:
        return ActionResult(action=action, status="queued", details={"queued_at": datetime.utcnow()})


class DecisionCoordinator:
    """Orchestrates ingestion -> reasoning -> approvals -> execution."""

    def __init__(
        self,
        connector: IngestionConnector,
        knowledge_graph: KnowledgeGraph,
        reasoning: ReasoningEngine,
        policy: PolicyEngine,
        executor: ActionExecutor,
    ) -> None:
        self._connector = connector
        self._kg = knowledge_graph
        self._reasoning = reasoning
        self._policy = policy
        self._executor = executor

    def run_daily_brief(self) -> Dict[str, Any]:
        signals = list(self._connector.fetch_signals())
        self._kg.ingest(signals)
        decision = self._reasoning.daily_brief()

        executed: List[ActionResult] = []
        approvals_needed: List[str] = []
        for action in decision.recommended_actions:
            if not self._policy.validate_action(action):
                continue
            if self._policy.requires_approval(action):
                approvals_needed.append(action)
            else:
                executed.append(self._executor.execute(action))

        return {
            "decision": decision,
            "signals_ingested": len(signals),
            "executed": executed,
            "approvals_needed": approvals_needed,
        }


def demo_run() -> Dict[str, Any]:
    """Quick demo wired with in-memory signals."""

    connector = InMemoryConnector(
        signals=[
            Signal(
                source="zendesk",
                timestamp=datetime.utcnow(),
                payload={"title": "CSAT drop in APAC", "metric": "csat", "delta": -8},
            )
        ]
    )
    kg = KnowledgeGraph()
    reasoning = ReasoningEngine(kg)
    policy = PolicyEngine()
    executor = ActionExecutor()
    coordinator = DecisionCoordinator(connector, kg, reasoning, policy, executor)

    return coordinator.run_daily_brief()


if __name__ == "__main__":
    output = demo_run()
    print("Decision summary:", output["decision"].summary)
    print("Approvals needed:", output["approvals_needed"])
    print("Executed actions:", [result.action for result in output["executed"]])
