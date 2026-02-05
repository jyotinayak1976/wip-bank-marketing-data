# Hila Starter Code (Enterprise Decision & Execution Layer)

This starter kit turns the strategy into an executable skeleton. It **does not** implement real integrations yet—it gives you a clean structure to plug them into.

## What this gives you

**One small file, five clear layers:**

1. **Ingestion + normalization** (`IngestionConnector`)
2. **Enterprise knowledge graph** (`KnowledgeGraph`)
3. **Reasoning + decision support** (`ReasoningEngine`)
4. **Policy + approvals** (`PolicyEngine`)
5. **Action / workflow execution** (`ActionExecutor`)

The orchestrator (`DecisionCoordinator`) ties them together for a daily brief.

## How to run the demo

```bash
python src/hila_starter.py
```

Example output:

```
Decision summary: Top risk: CSAT drop in APAC
Approvals needed: ['open_escalation']
Executed actions: ['notify_owner']
```

## Map the code to the product plan

| Product Layer | Code Class | What it does now | What you add next |
| --- | --- | --- | --- |
| Ingestion + normalization | `IngestionConnector` | Stubbed in-memory signals | Real connectors to CRM/ERP/tickets/warehouse |
| Knowledge graph | `KnowledgeGraph` | In-memory list of signals | Graph DB or KG service with relationship edges |
| Reasoning + decision support | `ReasoningEngine` | Simple heuristic | LLM + causal inference + business rules |
| Policy + guardrails | `PolicyEngine` | Hard-coded approvals | Configurable policies, auth, audit logs |
| Actions / workflows | `ActionExecutor` | Returns queued action | Real integrations + idempotency |

## Recommended next steps

1. **Pick the enterprise wedge** (e.g., CX Ops) and define the daily brief schema.
2. **Replace the in-memory connector** with one real system (Zendesk, Salesforce, or Freshdesk).
3. **Instrument outcomes** (e.g., CSAT uplift, escalation reduction).
4. **Add approvals + audit logs** before any write actions.

## Why this matters

This structure keeps you honest: you are not building “chat.” You are building **decision velocity + execution**, which is what enterprises actually buy.
