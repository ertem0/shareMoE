# Architecture

This document describes how the engine is built. [PROJECT.md](PROJECT.md) describes what is built and in which order.

## Overview

```text
                 Coordinator
 ┌──────────────────────────────────────────┐
 │ checkpoint store (full model on disk)    │
 │ dense runtime: embeddings, attention,    │
 │   routers, norms, LM head, KV cache      │
 │ dispatcher ──► expert directory          │
 │ node manager (sessions, health)          │
 │ request queue                            │
 └───────┬──────────────────────┬───────────┘
   control│  tensor       control│  tensor
         ▼                      ▼
   ┌────────────┐          ┌────────────┐
   │ Worker 1   │          │ Worker 2   │
   │ ExpertWorker│         │ ExpertWorker│
   │ experts... │          │ experts... │
   └────────────┘          └────────────┘
```

## Design principles

- **Model logic does not know about transport.** The dense runtime calls an expert executor interface. Whether an expert runs locally or on a worker is decided by the dispatcher.
- **Model-specific code lives in model adapters.** Weight names, expert module construction and dense layer construction for OLMoE live in one adapter. Everything else is model-agnostic.
- **`ExpertWorker` holds no I/O.** It registers modules and executes them. Networking and checkpoint access live outside it.
- **The coordinator is the single source of truth for expert ownership.** Workers never decide what they own.
- **Bulk data goes over the tensor connection.** The control connection carries only small metadata messages.
- **Fail explicitly.** Unknown IDs, malformed frames, timeouts and lost workers produce controlled errors. No silent fallbacks and no broad exception handling.
- **Tests are deterministic and offline.** Tests use small local fixtures and explicit synchronization, never Hugging Face access or sleeps.

## Components

### Coordinator

| Component        | Responsibility                                                                         |
| :--------------- | :------------------------------------------------------------------------------------- |
| Checkpoint store | Reads individual tensors from the local checkpoint directory given at startup          |
| Model adapter    | Builds dense components and expert modules from checkpoint tensors                     |
| Dense runtime    | Runs embeddings, attention, routers, norms and the LM head, and owns the KV cache      |
| Expert directory | Maps every expert to its owner node and status                                         |
| Allocator        | Assigns unallocated experts to a worker based on its memory budget                     |
| Node manager     | Accepts connections, holds node sessions, runs heartbeats and handles disconnects      |
| Dispatcher       | Groups routed tokens by owner node, sends requests concurrently and collects responses |
| Request queue    | Receives inference requests from any node and runs them one at a time                  |

### Worker

| Component      | Responsibility                                                                              |
| :------------- | :------------------------------------------------------------------------------------------ |
| Worker client  | Connects to the coordinator, runs initialization, answers heartbeats and handles disconnect |
| Model adapter  | Reconstructs expert modules from the metadata and weights the coordinator sends             |
| `ExpertWorker` | Registers expert modules and executes them by `(layer_id, expert_id)`                       |

### Shared

| Component    | Responsibility                                                   |
| :----------- | :--------------------------------------------------------------- |
| Framing      | Encodes and decodes frames                                       |
| Messages     | Defines message types and their metadata                         |
| Tensor codec | Serializes and deserializes CPU tensors                          |
| Connection   | Sends and receives frames with timeouts and records measurements |

## Expert directory

The directory stores one entry per `(layer_id, expert_id)` in the model, including unallocated experts.

```python
@dataclass
class ExpertEntry:
    layer_id: int
    expert_id: int
    status: ExpertStatus  # UNALLOCATED, RESERVED, LOADING, READY
    node_id: str | None
```

Internally it keeps two indexes that are always updated together.

- `by_expert: dict[tuple[int, int], ExpertEntry]` answers "who owns this expert?"
- `by_node: dict[str, set[tuple[int, int]]]` answers "what does this node own?" and makes releasing a node's experts cheap.

Required queries:

- `owner(layer_id, expert_id) -> str | None`
- `experts_of(node_id) -> set[tuple[int, int]]`
- `group_by_node(layer_id, expert_ids) -> dict[str, list[int]]`, used by the dispatcher for each layer
- `unallocated() -> list[tuple[int, int]]` in ascending order

All mutations go through one lock, so reservation, status changes and release are atomic.

## Allocation

1. All experts are assumed to have the same weight size in the loading dtype.
2. A worker reports `memory_budget_bytes` for expert weights only. The worker keeps memory outside that budget for loading and execution.
3. The coordinator reserves the first `min(unallocated_count, memory_budget_bytes // expert_size_bytes)` unallocated experts in ascending `(layer_id, expert_id)` order.
4. Reservation happens under the directory lock before any weights are sent, so two simultaneous requests can never receive the same expert.
5. Released experts become unallocated and are eligible for the next worker that initializes.

## Wire protocol

### Connections

The coordinator listens on two ports. Each worker opens two TCP connections.

- **Control connection:** initialization, assignments, load reports, heartbeats, inference requests and results, disconnect.
- **Tensor connection:** expert weights and expert execution requests and responses.

### Frame format

Both connections use the same frame.

```text
[length][message type][metadata length][metadata][payload]
```

| Field             | Encoding                                             |
| :---------------- | :--------------------------------------------------- |
| `length`          | uint64, big-endian, number of bytes after this field |
| `message type`    | uint16, big-endian                                   |
| `metadata length` | uint32, big-endian                                   |
| `metadata`        | UTF-8 JSON                                           |
| `payload`         | raw bytes, empty for control messages                |

A tensor payload holds the contiguous little-endian bytes of the tensor. Its `dtype` and `shape` are in the metadata. Frames with inconsistent lengths, unknown types or metadata that does not match the payload size are rejected.

### Message types

Control connection:

| Message                         | Direction            | Purpose                                      |
| :------------------------------ | :------------------- | :------------------------------------------- |
| `INIT_REQUEST`                  | worker → coordinator | Memory budget                                |
| `INIT_ACCEPT`                   | coordinator → worker | Node ID and session token                    |
| `ASSIGNMENT`                    | coordinator → worker | List of experts and reconstruction metadata  |
| `LOAD_REPORT`                   | worker → coordinator | Loaded experts and failed experts            |
| `ASSIGNMENT_CONFIRM`            | coordinator → worker | Final assignment after failures are released |
| `ASSIGNMENT_ACK`                | worker → coordinator | Worker accepts the final assignment          |
| `HEARTBEAT` / `HEARTBEAT_ACK`   | coordinator ↔ worker | Health monitoring                            |
| `INFERENCE_REQUEST`             | worker → coordinator | Prompt and generation settings               |
| `INFERENCE_RESULT`              | coordinator → worker | Generated result or error                    |
| `DISCONNECT` / `DISCONNECT_ACK` | either direction     | Clean disconnect                             |
| `ERROR`                         | either direction     | Controlled error with a code and message     |

Tensor connection:

| Message                | Direction            | Purpose                           |
| :--------------------- | :------------------- | :-------------------------------- |
| `ATTACH`               | worker → coordinator | Node ID and session token         |
| `ATTACH_ACK`           | coordinator → worker | Tensor connection accepted        |
| `EXPERT_WEIGHTS`       | coordinator → worker | Weights for one expert            |
| `EXPERT_EXEC_REQUEST`  | coordinator → worker | Hidden states and expert segments |
| `EXPERT_EXEC_RESPONSE` | worker → coordinator | Expert outputs                    |
| `EXPERT_EXEC_ERROR`    | worker → coordinator | Controlled execution error        |

### Expert execution request

The dispatcher sends one request per worker per layer. The payload holds the hidden states for every token routed to that worker, grouped by expert. A token routed to two experts on the same worker appears twice.

```json
{
  "request_id": "a1b2",
  "layer_id": 3,
  "dtype": "float32",
  "shape": [12, 2048],
  "segments": [
    { "expert_id": 5, "offset": 0, "count": 7 },
    { "expert_id": 41, "offset": 7, "count": 5 }
  ]
}
```

The response uses the same `request_id`, shape and segment layout. The coordinator maps rows back to tokens and applies the router weights. Workers never see router weights.

## Sequences

### Initialization

1. Worker connects to the control port and sends `INIT_REQUEST`.
2. Coordinator reserves experts, then replies `INIT_ACCEPT`.
3. Worker connects to the tensor port and sends `ATTACH`. The coordinator rejects unknown tokens and duplicate attachments, and applies an attach timeout.
4. Coordinator sends `ASSIGNMENT` on the control connection, then one `EXPERT_WEIGHTS` per expert on the tensor connection.
5. Worker reconstructs and registers each expert and sends `LOAD_REPORT`.
6. Coordinator resends failed experts, up to three attempts per expert in total. Experts that still fail are released.
7. Coordinator sends `ASSIGNMENT_CONFIRM` with the final assignment. Worker replies `ASSIGNMENT_ACK`.
8. Coordinator marks the worker and its experts ready.

Connections that have not sent `INIT_REQUEST` or `ATTACH` yet are held in a pending set with a timeout. Public snapshots of node state never include sockets or session tokens.

### Inference

1. A request arrives on the coordinator's queue, from the coordinator itself or as an `INFERENCE_REQUEST` from a worker.
2. If any expert is not ready, the request fails with an explicit error.
3. The dense runtime runs prefill, then decodes token by token. For every layer the dispatcher:
   1. takes the router's top-k experts and weights per token,
   2. calls `group_by_node` on the expert directory,
   3. sends one `EXPERT_EXEC_REQUEST` per involved worker concurrently,
   4. waits for all responses, with a timeout,
   5. returns the weighted sum to the dense runtime.
4. After the final token, the result goes back to the requester, locally or as `INFERENCE_RESULT`.

### Health monitoring

The coordinator sends `HEARTBEAT` at a fixed interval. If no `HEARTBEAT_ACK` arrives within the timeout, or either connection fails, the coordinator:

1. closes both connections,
2. releases all of the worker's experts, including reservations,
3. fails every in-flight request that involves the worker.

### Clean disconnect

Started by the coordinator:

1. Coordinator stops dispatching to the worker and waits for its in-flight requests.
2. Coordinator sends `DISCONNECT`. Worker replies `DISCONNECT_ACK`.
3. Both sides close both connections. The coordinator releases the worker's experts.

Started by the worker:

1. Worker sends `DISCONNECT`.
2. Coordinator stops dispatching to the worker, waits for its in-flight requests and releases its experts.
3. Coordinator replies `DISCONNECT_ACK` and both sides close both connections.

## Project structure

```text
src/moe_engine/
├── experts/
│   ├── interface.py      # expert execution interface
│   └── worker.py         # ExpertWorker
├── models/
│   ├── adapter.py        # model adapter interface
│   └── olmoe.py          # OLMoE adapter
├── checkpoint/
│   └── loader.py
├── protocol/
│   ├── framing.py
│   ├── messages.py
│   └── tensors.py
├── transport/
│   └── connection.py
├── coordinator/
│   ├── server.py
│   ├── directory.py
│   ├── allocation.py
│   ├── health.py
│   ├── dispatch.py
│   ├── runtime.py        # dense runtime and KV cache
│   └── queue.py
├── worker/
│   └── client.py
└── bench/
tests/
├── unit/
└── integration/
```
