# Architecture: Plan → Execute → Validate

## Overview

The PEV graph extends the standard LangGraph plan-and-execute pattern with a
third node — a structured **Validator** — plus a **Router** that implements
deterministic retry and replanning logic. The result is an agent loop with
explicit quality gates and a full audit trail.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    User(["👤 Caller"])

    subgraph PEVGraph["PEV Graph  (LangGraph StateGraph)"]
        direction TB

        Planner["🧠 Planner\nclaude-haiku\nStructured JSON output"]
        Executor["⚙️ Executor\nclaude-sonnet\nTool-call loop"]
        Validator["🔍 Validator\nclaude-haiku\nConfidence score 0–1"]
        Router["🔀 Router\nPure Python\nNo LLM"]

        Planner --> Executor
        Executor --> Validator
        Validator --> Router
    end

    Tools["🛠️ Tools\n(Tavily, custom, etc.)"]
    LangSmith["📊 LangSmith\nObservability"]

    User -->|"initial_state(task)"| Planner
    Router -->|"score ≥ threshold\nnext step"| Executor
    Router -->|"retry_count < max"| Executor
    Router -->|"retry exhausted"| Planner
    Router -->|"complete / failed"| User
    Executor <-->|"tool calls"| Tools
    PEVGraph -.->|"traces"| LangSmith
```

---

## 2. State Machine

```mermaid
stateDiagram-v2
    [*] --> Planning : graph.invoke()

    Planning --> Executing : plan generated

    Executing --> Validating : step complete

    Validating --> Executing : score ≥ threshold\nmore steps remain
    Validating --> Complete  : score ≥ threshold\nfinal step
    Validating --> Executing : score < threshold\nretry_count < max_retries
    Validating --> Planning  : score < threshold\nretry_count ≥ max_retries\nreplan_count < max_replans
    Validating --> Failed    : all limits exhausted

    Complete --> [*] : status = complete
    Failed   --> [*] : status = failed
```

---

## 3. Request Lifecycle — Happy Path

```mermaid
sequenceDiagram
    actor U  as Caller
    participant G  as Graph
    participant P  as Planner (haiku)
    participant E  as Executor (sonnet)
    participant T  as Tools
    participant V  as Validator (haiku)
    participant R  as Router

    U->>G: graph.invoke(initial_state(task))
    G->>P: task
    P-->>G: plan [step1, step2, step3]

    loop For each step
        G->>E: current_step + context
        E->>T: tool_call(args)
        T-->>E: tool result
        E-->>G: pending_result

        G->>V: step + pending_result + task
        V-->>G: score=0.92, feedback="Complete."

        G->>R: score=0.92, retry=0, idx=n
        R-->>G: _next="execute", idx=n+1
    end

    G->>R: score=0.88, idx=2 (last step)
    R-->>G: _next="complete"
    G-->>U: PEVState(status="complete", step_results=[...])
```

---

## 4. Retry Flow

```mermaid
sequenceDiagram
    participant E  as Executor (sonnet)
    participant V  as Validator (haiku)
    participant R  as Router

    E-->>V: pending_result (attempt 1)
    V-->>R: score=0.55, feedback="Missing X"

    R-->>E: retry (retry_count=1)\nfeedback injected into prompt

    E-->>V: pending_result (attempt 2, improved)
    V-->>R: score=0.85, feedback="Good."

    R-->>E: advance to next step
```

---

## 5. Replan Flow

```mermaid
sequenceDiagram
    participant E  as Executor
    participant V  as Validator
    participant R  as Router
    participant P  as Planner

    E-->>V: pending_result (attempt 1)
    V-->>R: score=0.40

    R-->>E: retry (retry_count=1)
    E-->>V: pending_result (attempt 2)
    V-->>R: score=0.35

    R-->>E: retry (retry_count=2)
    E-->>V: pending_result (attempt 3)
    V-->>R: score=0.38  ← retries exhausted

    R-->>P: replan (replan_count=1)\nfeedback + past steps injected
    P-->>E: revised plan

    E-->>V: pending_result (revised step 1)
    V-->>R: score=0.91 ✓
```

---

## 6. Router Decision Tree

```mermaid
flowchart TD
    A["Router receives\nvalidation_score, retry_count,\nreplan_count, current_step_idx"]

    B{"score ≥\npass_threshold?"}
    C{"Last step\nin plan?"}
    D["✅ complete\nroute → END"]
    E["➡️ execute\nadvance idx\nreset retry_count"]

    F{"retry_count <\nmax_retries?"}
    G["🔁 retry\nincrement retry_count\nroute → executor"]

    H{"replan_count <\nmax_replans?"}
    I["🔄 replan\nroute → planner"]
    J["❌ failed\nset error message\nroute → END"]

    A --> B
    B -->|yes| C
    C -->|yes| D
    C -->|no| E
    B -->|no| F
    F -->|yes| G
    F -->|no| H
    H -->|yes| I
    H -->|no| J
```

---

## 7. State Schema

```mermaid
classDiagram
    class PEVState {
        +str task
        +list~str~ plan
        +int current_step_idx
        +str pending_result
        +list~StepResult~ step_results
        +float validation_score
        +str validation_feedback
        +int retry_count
        +int replan_count
        +Status status
        +str|None error
        +str _next
    }

    class StepResult {
        +str step
        +str result
        +float score
        +str feedback
        +int attempts
    }

    class PEVConfig {
        +str planner_model
        +str executor_model
        +str validator_model
        +float pass_threshold
        +int max_retries
        +int max_replans
        +list~BaseTool~ tools
    }

    PEVState "1" --> "*" StepResult : step_results (operator.add)
    PEVConfig --> PEVState : configures
```

---

## 8. Cost Model — Why Three Models

```mermaid
graph LR
    subgraph Cheap["Cheap Model  (~$0.25/1M tokens)"]
        P["Planner\nStructured JSON only\nNo reasoning required"]
        V["Validator\nScore + one sentence\nNo reasoning required"]
    end

    subgraph Capable["Capable Model  (~$3/1M tokens)"]
        E["Executor\nTool calls + multi-step reasoning\nQuality matters here"]
    end

    P -->|"generates"| Plan["plan: list[str]"]
    E -->|"produces"| Result["pending_result: str"]
    V -->|"scores"| Score["score: float\nfeedback: str"]
```

The planner and validator only produce structured JSON — a cheap model
handles this perfectly. The executor is where reasoning and tool use happen;
investing in a capable model here drives the quality of the final output.

**Typical cost split per run (3-step task):**
- Planner: ~500 tokens × haiku rate = ~$0.0001
- Executor: ~3,000 tokens × sonnet rate = ~$0.009
- Validator: ~1,500 tokens × haiku rate = ~$0.0004
- **Total: ~$0.01 per run** vs ~$0.027 if using sonnet for all three

---

## 9. Tool-Call Loop (Executor)

```mermaid
flowchart TD
    Start["Executor invoked\nwith step + context"]
    SendToLLM["Send to LLM\n(tools bound)"]
    HasToolCalls{"Response has\ntool_calls?"}
    ExecuteTools["Execute each tool\ncollect results"]
    FeedBack["Append ToolMessage\nto messages"]
    CapCheck{"round ≥\nMAX_TOOL_ROUNDS?"}
    ExtractText["Extract final\ntext content"]
    WritePending["Write to\npending_result"]

    Start --> SendToLLM
    SendToLLM --> HasToolCalls
    HasToolCalls -->|yes| ExecuteTools
    ExecuteTools --> FeedBack
    FeedBack --> CapCheck
    CapCheck -->|no| SendToLLM
    CapCheck -->|yes| ExtractText
    HasToolCalls -->|no| ExtractText
    ExtractText --> WritePending
```

The loop is capped at `MAX_TOOL_ROUNDS = 10` to prevent runaway agents.
Unknown tool names produce an error string (not an exception) so the LLM
can recover gracefully.

---

## 10. Audit Trail

Every execution attempt is preserved in `step_results` via `operator.add`.
This means retried steps produce multiple entries — full history, nothing
overwritten.

```
step_results = [
    StepResult(step="Search for X",  score=0.55, attempts=1, feedback="Missing Y"),
    StepResult(step="Search for X",  score=0.88, attempts=2, feedback="Good."),
    StepResult(step="Summarise X",   score=0.92, attempts=1, feedback="Complete."),
    StepResult(step="Write report",  score=0.95, attempts=1, feedback="Excellent."),
]
```

This audit trail is the core operational signal: you can see exactly where
the agent struggled, what feedback it received, and how many attempts each
step took.
