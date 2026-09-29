# Distributed MoE Inference Engine

## Goal

Build an inference engine for Mixture-of-Experts (MoE) language models that distributes the experts across worker nodes, so that no single device needs enough memory for the whole model.

Expert weights account for most of the parameters in an MoE model, while each token only uses a few experts. The coordinator keeps the non-expert parts of the model and sends hidden states to whichever workers hold the experts the router selects.

## Research question

What does remote expert execution cost in communication time relative to expert compute time, and how does that cost change with the number of tokens, the tensor size and the number of workers?

## Node roles

### Coordinator

- There is one coordinator per deployment. This is not enforced explicitly, because every worker connects to a single coordinator address.
- Holds the full model checkpoint on local storage.
- Hosts and computes every non-expert component: embeddings, attention, routers, normalization layers and the LM head.
- Initializes workers by allocating experts to them and sending those experts' weights over TCP.
- Keeps an expert directory that answers "which node holds expert `(layer_id, expert_id)`?"
- Tracks the health of every worker.
- Sends hidden-state tensors to workers for expert computation and receives the results.
- Accepts inference requests from itself or from any worker and returns the result to the requester.
- Disconnects workers cleanly.

### Worker

- Connects to the coordinator and requests experts, reporting the memory it can dedicate to expert weights.
- Receives, reconstructs and serves the experts assigned to it.
- Receives a hidden-state tensor together with the expert(s) to apply, computes the result and returns it to the coordinator.
- Can submit inference requests to the coordinator.
- Disconnects cleanly from the coordinator.

Each worker keeps two TCP connections to the coordinator, a control connection and a tensor connection. See [ARCHITECTURE.md](ARCHITECTURE.md#wire-protocol) for the protocol.

## Inference flow

1. A node (the coordinator itself or any worker) sends a prompt to the coordinator.
2. The coordinator tokenizes the prompt and computes the embeddings.
3. For each layer, starting at layer 0:
   1. The coordinator computes attention and the router for the layer.
   2. The coordinator selects the top-k experts for each token and looks up which worker holds each selected expert.
   3. The coordinator sends each involved worker one request with the hidden states for the tokens routed to that worker's experts.
   4. Each worker computes its experts and returns the outputs.
   5. The coordinator combines the expert outputs with a weighted sum using the router weights and continues to the next layer.
4. After the last layer, the coordinator applies the final normalization and the LM head and samples the next token.
5. Steps 3 and 4 repeat for each generated token until an end-of-sequence token or the token limit is reached.
6. The coordinator returns the generated result to the node that made the request.

## V1 scope

- First supported model: `allenai/OLMoE-1B-7B-0924`. The engine itself must not be specific to this model. Model-specific code lives in a model adapter.
- Topology: one coordinator and two workers. Development and tests run as local processes on one machine. The final V1 milestone runs on separate machines.
- The coordinator processes one inference request at a time. Other requests wait in a queue.
- Tensors sent over the network are CPU tensors in float32, float16 or bfloat16.
- Inference requires every expert to be allocated to a ready worker. Requests are rejected with an explicit error otherwise.

## Non-goals

- Training or fine-tuning.
- Production-scale serving.
- More than one coordinator.

## Milestones

Each milestone has a status (`not started`, `in progress` or `done`). Only one milestone should be `in progress` at a time.

### Milestone 1: Local expert execution

Status: in progress

Scope:

- Define a model-agnostic expert execution interface.
- Identify experts by `(layer_id, expert_id)`.
- Implement an expert module with OLMoE's expert shapes and random weights for testing.
- Execute a selected expert locally.

Done when:

- A test selects an expert, executes it locally and gets the expected output shape.
- Repeated execution with the same input and weights gives the same output.
- The interface has no dependency on networking.

### Milestone 2: Checkpoint download and component loading

Status: not started

Scope:

- Configurable Hugging Face model ID, pinned revision and local checkpoint directory.
- Download the checkpoint to the coordinator's storage once and reuse it on later starts.
- Load a single expert by `(layer_id, expert_id)` without building the whole model in memory.
- Load the non-expert components separately from the experts.
- Keep model-specific weight extraction in the model adapter, separate from download logic.
- Fail explicitly on unknown experts, missing weights, incompatible metadata or download failures.

Done when:

- The checkpoint downloads to the configured directory and is reused on a later start.
- A loaded expert matches reference execution with the same weights and inputs within floating-point tolerance.
- Loading works offline once the checkpoint files are present.
- Tests use small local fixtures and need no Hugging Face access.

### Milestone 3: Expert worker

Status: not started

Scope:

- Implement `ExpertWorker`, which registers already constructed expert modules and executes them by `(layer_id, expert_id)`.
- Support executing several experts in one call, each on its own slice of rows.
- `ExpertWorker` performs no checkpoint loading, downloads or networking.

Done when:

- A worker hosting several experts executes the correct expert for each ID.
- A multi-expert call matches the same experts executed one at a time.
- An unknown expert ID fails with a controlled error.
- All tests run without networking.

### Milestone 4: Wire protocol and tensor transport

Status: not started

Scope:

- Implement the frame format `[length][message type][metadata length][metadata][payload]`.
- Serialize and deserialize CPU tensors of any shape in float32, float16 and bfloat16.
- Provide independent send and receive operations, with request and response built on top.
- Apply timeouts and fail explicitly on malformed frames, unsupported tensors, disconnects and timeouts.
- Record timing and size measurements: serialization, deserialization, payload bytes, framed bytes, socket send duration and round-trip duration.
- Keep the transport independent of model-specific logic.

Done when:

- One-way and echo transfers between two local processes preserve shape, dtype and values.
- Tests cover every supported dtype, arbitrary shapes, empty tensors, noncontiguous inputs and malformed frames.
- Measurements are recorded for every transfer.

### Milestone 5: Remote expert execution

Status: not started

Scope:

- Send an expert execution request over the tensor connection to a worker process whose experts were registered directly, without the initialization protocol.
- A request can target several experts on the same worker.
- The worker returns the outputs, or a controlled error.

Done when:

- Remote output matches local execution within floating-point tolerance.
- A request for an unknown expert returns a controlled error.
- Model logic stays separate from transport logic.

### Milestone 6: Expert directory

Status: not started

Scope:

- Implement the coordinator's expert directory described in [ARCHITECTURE.md](ARCHITECTURE.md#expert-directory).
- Track every `(layer_id, expert_id)` with its owner node and status.
- Query the owner of an expert, the experts of a node, and group a set of selected experts by owner node.
- Reserve experts atomically so two workers can never own the same expert.

Done when:

- Unit tests cover lookup, grouping by node, reservation without duplicate ownership and releasing all experts of a node.

### Milestone 7: Worker initialization and expert distribution

Status: not started

Scope:

- A worker opens the control connection and sends an initialization request with its memory budget.
- The coordinator allocates experts following the allocation rule in [ARCHITECTURE.md](ARCHITECTURE.md#allocation), issues a node ID and session token, and the worker attaches its tensor connection.
- The coordinator sends the assignment list and the expert weights. The worker reconstructs and registers the experts.
- The worker reports success or lists the experts that failed to load.
- Failed experts are resent, with at most three attempts per expert. Successfully loaded experts are kept.
- Experts that still fail are released, and the coordinator confirms the reduced assignment before marking the worker ready.
- Every coordinator start sends all weights again. Workers never download the model.

Done when:

- Allocation assigns the first unallocated experts that fit the budget, with no duplicate ownership.
- A worker executes its assigned experts and matches local execution within floating-point tolerance.
- Retries target only failed experts and stop after three attempts.
- A worker is marked ready only after its final assignment is loaded and its tensor connection is attached.
- Tests use small local fixtures and run between local processes without Hugging Face access.

### Milestone 8: Health monitoring and clean disconnect

Status: not started

Scope:

- Heartbeats on the control connection with a configurable interval and timeout.
- On timeout or failure of either connection, close both connections, release all of that worker's experts and fail any in-flight requests to it.
- Clean disconnect started by either side, following the sequence in [ARCHITECTURE.md](ARCHITECTURE.md#clean-disconnect).

Done when:

- A killed worker's experts become unallocated within the timeout.
- A clean disconnect from either side leaves no open sockets and no assignments for that worker.
- In-flight requests to a lost worker fail with an explicit error instead of hanging.

### Milestone 9: Distributed MoE layer

Status: not started

Scope:

- The coordinator computes attention and the router for one layer.
- Select the top-k experts per token and group tokens by owner node using the expert directory.
- Send one request per involved worker, concurrently, and wait for all responses.
- Combine expert outputs with the router weights and restore the original token order.

Done when:

- One OLMoE MoE layer with its experts spread over two workers matches a fully local reference within floating-point tolerance.
- Tests cover multiple tokens, multiple experts per token and a token whose experts sit on different workers.

### Milestone 10: End-to-end distributed inference

Status: not started

Scope:

- Run the full forward pass across all layers, including prefill and token-by-token decoding, with the KV cache on the coordinator.
- Accept inference requests from the coordinator and from any worker over the control connection, and return the result to the requester.
- Queue requests and process them one at a time.
- Reject requests while any expert is unallocated.
- Run the coordinator and the workers on separate machines.

Done when:

- Greedy generation in float32 produces the same tokens as a fully local reference for a fixed prompt.
- A worker can submit a prompt and receive the generated result.
- A request completes with the coordinator and two workers on separate machines.

### Milestone 11: Benchmarking

Status: not started

Scope:

- Measure per remote call: serialization, deserialization, request and response bytes, request transmission, expert compute and response transmission.
- Measure total remote and local expert invocation time, per-layer time and tokens per second.
- Report averages and p50, p95 and p99 latency.
- Compare local and remote execution while varying the number of tokens and the number of workers.
- Save results to files for later analysis.

Done when:

- A benchmark run compares local and remote expert execution with each latency component reported separately.
- Results are saved and can be reloaded.
- Enabling measurement does not change the computed outputs.

## V1 completion

V1 is complete when Milestones 1 to 11 are done. The result should show that the experts of an MoE model can live on other nodes, be used transparently during inference, and be evaluated for correctness and communication cost.

## Future scope

Not required for V1. Do not implement unless a task explicitly asks for it.

- Redistributing experts automatically after a worker is lost.
- Persistent expert caches on workers and incremental weight transfers.
- Processing several inference requests concurrently or batching across requests.
- Replicating experts on more than one worker.
- Load-aware or latency-aware expert placement.
- Coordinator-local expert execution as a fallback.
- GPU-to-GPU transfers.
- Authentication and encryption beyond the session token.
