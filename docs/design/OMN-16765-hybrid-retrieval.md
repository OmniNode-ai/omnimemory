<!--
SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
SPDX-License-Identifier: MIT
-->

# OMN-16765 — Hybrid vector + full-text retrieval with rank fusion

**Status:** draft, for review
**Author:** Lakshman Patel
**Ticket:** [OMN-16765](https://linear.app/omninode/issue/OMN-16765)
**Verified against:** `dev` @ `1c84840`

This is the Step 1 design note required by §4 of the ticket. It must be reviewed
before implementation code is written.

---

## 1. Placement — settled

Fused ranking is built **in `omnimemory`**, in
`src/omnimemory/nodes/node_memory_retrieval_effect/handlers/`, beside the
existing handler family. Nothing is added to `omnimarket`.

The governing document is
[omnimemory-market-migration-boundary.md](https://github.com/OmniNode-ai/knowledge-base/blob/main/guides/omnimemory-market-migration-boundary.md)
(last verified 2026-08-25). Note it is **more precise than §3 of the ticket**:
§3 summarises the rule as "adapters stay, handlers move", but the doc records
that node-local models, `contract.yaml`, clients, registry, utils and validators
move too — only each node's `adapters/` subdirectory stays. Where the two
disagree, this note follows the doc.

Confirmed locally:

- `omnimarket/.../node_memory_retrieval_effect/` has no `handlers/` directory.
- Its `contract.yaml:18` resolves the handler module into the `omnimemory`
  package, and `omnimarket/pyproject.toml:76` pins `omninode-memory==0.17.1` —
  a real published-package dependency, not a path shim.
- `node_memory_retrieval_effect` is a Wave 2 node; canonical side is recorded as
  "not verified per-node". 13 of 15 nodes still exist on both sides.

**Why this is right rather than merely convenient.** Fusion consumes the
*results* of two backend legs and reorders them. It opens no connection and
implements no storage protocol, so under the boundary doc's own split it is in
the "moves" column — and the "moves" column physically lives in `omnimemory`
today. Splitting the handler family across repos mid-migration would turn a
mechanical cutover into a redesign: `HandlerMemoryRetrieval` fans out to its
sub-handlers in-process, so fusion landing in `omnimarket` creates a cross-repo
call inside a single operation.

**Model placement.** Any new model goes in the node's `models/` directory. This
is compatible with the repo invariant, not an exception to it:
`scripts/validation/validate_model_locations.py:68` lists
`nodes/node_memory_retrieval_effect/models/` in `ALLOWED_PATHS`, so no
`# omnimemory-model-exempt:` marker is needed.

---

## 2. Surface — a new `search_hybrid` operation

Per §3 of the ticket ("do not introduce a parallel retrieval entry point") and
§6.1 of the architecture plan ("new operation `search_hybrid`; existing `search`
and `search_text` unchanged"), fusion is added as a fourth operation behind the
existing protocol. Existing callers are untouched.

Adding an operation is a four-point change, and the contract records what
happens when it is done partially. `contract.yaml` carries a long comment
explaining that the `index` operation was **removed rather than repointed**,
because `ModelMemoryRetrievalRequest.operation` is
`Literal["search", "search_text", "search_graph"]` and
`HandlerMemoryRetrieval.execute()` closes its match with
`assert_never(request.operation)` — so a declared route the Literal cannot
express is a route the type system forbids. `search_hybrid` therefore changes
all four points together:

| # | File | Change |
|---|---|---|
| 1 | `models/model_memory_retrieval_request.py:97` | extend the `Literal` with `"search_hybrid"` |
| 2 | `models/model_memory_retrieval_request.py` validator | require `query_text` for the new operation |
| 3 | `handlers/handler_memory_retrieval.py` | add `case "search_hybrid"` before `assert_never` |
| 4 | `contract.yaml` `operations:` | declare the operation, its `inputs`/`outputs` |
| 5 | `contract.yaml` `handler_routing.handlers:` | declare the route |
| 6 | `contract.yaml` `constraint_definitions.operation` | extend the mirrored `Literal` string |
| 7 | `handlers/handler_fusion.py` | new — the ranking functions |

**`contract.yaml` needs three separate edits, not one.** `operations:` (~line 70)
describes the operation, `handler_routing.handlers:` (~line 220) binds it to a
handler, and `constraint_definitions.operation` (~line 173) restates the DTO's
`Literal` as a string that must stay in sync.

Point 6 is the easiest to miss and is guarded:
`TestHandlerRoutingKeyAlignment::test_routing_keys_align_with_constraints`
derives the legal routing keys from that string and fails with
`Unexpected routing_keys: {'search_hybrid'}` if the route is declared without
it. That guard caught this during implementation — worth knowing about before
adding any future operation, because the failure names the route rather than
the constraint line, which points at the wrong file.

One apparent counterexample, resolved so a reviewer does not have to: the
contract declares a `health_check` operation that is *not* in the request
`Literal`. That is not the `index` defect repeating — `health_check` declares
`inputs: []` and appears identically in five node contracts; it is a
platform-level operation that never constructs a `ModelMemoryRetrievalRequest`,
so the Literal has nothing to express. `search_hybrid` does take a request, so
it does need all five points.

Fusion lives in its own module rather than as a method on
`HandlerMemoryRetrieval`. The rank-fusion function is pure — it maps two ranked
lists to one — and keeping it free of handler lifecycle makes it directly
testable, which is what the fixture-corpus test in §5 needs.

`search_hybrid` fans out to the existing `search` and `search_text` legs
concurrently via `asyncio.gather` (the pattern `initialize()` already uses at
`handler_memory_retrieval.py:151`), then fuses.

**`search_graph` is deliberately not a third leg.** Plan §3.1.3 scopes hybrid to
vector + FTS, and graph traversal answers a structurally different question —
it starts from a `snapshot_id` rather than a query string, so it has no ranked
list of query-relevant documents to contribute to a rank fusion. It stays its
own operation, unchanged.

---

## 3. Lexical backend — PostgreSQL `tsvector`

**Chosen:** Postgres full-text search, reached through the existing `search_text`
leg.

**Framing note, per the ticket's own correction.** The motivating claim — that
vector-only retrieval fails on exact-token queries — is a general property of
dense retrieval, **not a measured symptom of a running system here**. §1 of the
ticket now records that there is no live retrieval ranking today, vector-only or
otherwise; what exists is the mock path. Nothing in this note should be read as
citing a production observation, and the fixture corpus in §5 is what turns the
general claim into something measured.

**Why:** the contract already declares a `search_text` route to a Postgres
full-text handler, and the architecture plan's §3.1.3 explicitly adopts the
fusion pattern while rejecting MemOS's SQLite FTS5 engine — *"we use Postgres
`tsvector` rather than SQLite FTS5. The fusion algorithm is the valuable
pattern, not the specific FTS engine."* Reusing the declared leg means fusion
adds no new backing service and no new dependency.

**Rejected:**

- **SQLite FTS5** — the upstream MemOS choice. Rejected: it would add a second
  storage engine to a service that already has Postgres declared for exactly
  this purpose.
- **Qdrant sparse vectors / BM25 inside Qdrant** — would put both legs in one
  backend and avoid a second round trip. Rejected: it collapses the two signals
  into one engine's scoring, which is precisely what makes the fusion function
  untestable in isolation. It also abandons the already-declared `search_text`
  route.
- **In-process lexical scoring (rank_bm25 or similar)** — attractive because it
  needs no service at all. Rejected as the *production* answer: it duplicates
  ranking logic Postgres already implements and would drift from it. See §5 for
  why the *test* deliberately does something adjacent to this.

---

## 4. Fusion function — RRF, `k = 60`, with the reasoning stated

**Chosen:** Reciprocal Rank Fusion, `score(d) = Σ 1 / (k + rank_i(d))`, `k = 60`.

The architecture plan's §3.1.3 names RRF with `k = 60`, and jonah's ticket
comment is explicit that this is *"a default, not a ruling"* — §4 of the ticket
invites an alternative if it buys something. I am accepting the default, and the
reason is a property of the inputs rather than deference:

RRF consumes **ranks, not scores**. The two legs here produce numerically
incomparable quantities — Qdrant returns cosine similarity in `[0, 1]`, Postgres
`ts_rank` returns an unbounded relevance float whose magnitude depends on
document length and term frequency normalisation. Any score-combining
alternative (weighted sum, CombSUM, z-score normalisation) requires first
mapping those onto a shared scale, and that mapping is an extra tuned parameter
with no principled value — it would have to be fitted on the same fixture corpus
used to evaluate the result, which makes the evaluation circular.

`k = 60` is the value from the original Cormack et al. RRF paper and the plan.
It is a smoothing constant: larger `k` flattens the contribution of top ranks,
smaller `k` sharpens it. I have no evidence-based reason to move it, and picking
a different number without evidence would be worse than taking the documented
default. The fixture corpus in §5 makes `k` a tunable we can revisit with
numbers later.

**Deduplication.** Fusion merges on `snapshot_id`. A document returned by both
legs accumulates both reciprocal-rank terms — that accumulation is the whole
mechanism by which agreement between legs is rewarded.

### What score the caller gets back

Results carry the **fused** score, normalised against the top hit — not the
score the originating leg gave them.

This is correctness, not presentation. Carrying each leg's own number through
lets the score and the ordering contradict each other: a document promoted
*because both legs agreed on it* can sit above one with a higher raw cosine, so
a caller sorting by `score` would silently reorder the very ranking this
operation exists to produce. Found in review on
[omnimemory#459](https://github.com/OmniNode-ai/omnimemory/pull/459);
`test_scores_never_contradict_the_returned_ordering` is the guard.

`fuse_rrf` is therefore split into `fuse_rrf_scores` (accumulate) and `fuse_rrf`
(order), so the handler draws the score it attaches and the order it returns
from a single source and they cannot drift.
`test_fuse_rrf_scores_and_fuse_rrf_cannot_diverge` guards that.

Normalisation is against the maximum because `ModelSearchResult.score` is
declared `ge=0.0, le=1.0` and documented as a relevance score. Raw RRF values
are ~0.016–0.033: they satisfy the bound while reporting a perfect match as "3%
relevant". Dividing by the top score is monotonic, so ordering is untouched and
only the scale changes.

**This is not the score normalisation rejected above, and the distinction
matters.** §4 rejects normalising the two legs' scores onto a shared scale *as
an input to fusion* — that would be a fitted parameter with no principled value,
and it would have to be fitted on the same corpus used to evaluate the result,
which is circular. What happens here is a monotonic rescale of a single
already-fused quantity on the way out: it changes no ranking decision and
introduces no tunable. The first would corrupt the measurement; the second
cannot.

The consequence for callers is recorded on `ModelSearchResult`: a hybrid score
is comparable **within one response only** — never across responses, and never
against a raw backend score from `search` or `search_text`.

---

## 5. Measurement — deterministic fixture corpus, NDCG@10

This section reflects jonah's Q3 ruling, which supersedes the ticket's own
wording. Recording the conflict explicitly so the choice is auditable:

> §4 step 2 and §5 of the ticket both call for an **integration test**, and §1
> frames the work as *"provable entirely against the repo's local
> `docker-compose` services"*. The Q3(b) ruling instead requires the measurement
> be *"deterministic and runnable in CI — no live Qdrant, no live Postgres, no
> network."*

**The ruling wins, and CI corroborates it.** `.github/workflows/ci.yml`'s test
job starts no service containers and runs no `docker-compose`; it selects test
paths and shards them with `pytest-split`. A test requiring live Qdrant or
Postgres would error in the merge gate, not skip. The deterministic design is
not merely preferable — it is the only shape that is actually gated.

**Marker:** `@pytest.mark.unit`. `pyproject.toml` defines `unit` as *"no
external dependencies"* and `integration` as *"may require external services"*,
and `--strict-markers` is on. A no-network fixture test is `unit` by the repo's
own definition, whatever §4/§5 call it.

**Metric:** **NDCG@10**, asserted against a numeric threshold.

Chosen over precision@k because the corpus has graded relevance, not binary —
for an exact-token query the exactly-matching document and a semantically
adjacent one are both non-useless, and they should not score identically.
Precision@k discards rank order within the cut entirely, which is the wrong
instrument for evaluating a *ranking* change: a fusion that moves the right
answer from position 8 to position 1 scores identically under precision@10 and
visibly better under NDCG@10. `k = 10` because the plan's own success criterion
(§7) is framed on top-3 behaviour, and 10 gives headroom to see movement below
the cut without rewarding a long tail.

**Corpus size — 24, not the plan's 50, deliberately.** Plan §7 sets a benchmark
of *"50 known error → resolution pairs"*. That 50 was sized for a **production
precision benchmark over real error/resolution data**, which is a different
instrument from this one: it measures deployed retrieval, and it is the shape
AC5 on OMN-16928 will need.

This corpus tests a **fusion function**. Its discriminating power comes from
covering the cases where the two legs disagree, not from volume — 50
hand-authored *synthetic* queries would be padding, and padding a labelled set
makes it look more authoritative than it is. 24 queries, 8 per family, is
enough for every family assertion to be a mean over 8 independent cases.
Extending is mechanical if the number is ever wanted.

Flagging the deviation rather than quietly matching the plan's figure, since
§4 asks for the corpus to be justified.

**Corpus shape.** Labels are graded (`2` = exact answer, `1` = related,
`0` = irrelevant), checked into the repo beside the fixtures. Three families,
and the third is the one that keeps the test honest:

1. **Exact-token queries** — identifiers, error codes, file names, quoted
   strings. Lexical leg right, vector leg wrong.
2. **Paraphrase queries** — semantically equivalent, no shared tokens. Vector
   leg right, lexical leg misses entirely. These prove fusion does not *regress*
   the semantic case.
3. **Agreement queries** — both legs return the same document highly ranked.
   These prove fusion does not *disturb* cases that already worked.

§4 of the ticket warns that a corpus where both approaches score identically
proves nothing. Family 1 alone would produce a fusion that only ever helps,
which is usually a fusion that was never tested on the case where it hurts —
so the gate asserts on all three, and family 2's threshold is a
**non-regression** bound against the vector-only arm rather than an improvement
bound.

**Where the two arms come from.** With no live backends, both the vector-only
baseline and the lexical leg need a deterministic source. Pre-computed
per-query results — document IDs with their similarity scores and `ts_rank`
values — are checked into the fixture alongside the labels. Both arms are then
reproducible from one file, and the test measures only the fusion function,
which is the variable this ticket actually controls.

This is deliberate rather than a workaround: a live comparison would confound
fusion quality with backend behaviour (embedding-server variance, index state,
text-search configuration, ingestion drift). If the number moved you could not
say whether the fusion improved or the index changed. The live comparison is
**AC5 on [OMN-16928](https://linear.app/omninode/issue/OMN-16928)**, deliberately
not collapsed into this one: this test proves the design, that one proves the
deployment.

---

## 6. Activation decay — parameterisation, and one thing I cannot decide locally

Per §4 step 4, decay lands as a **separate PR after hybrid search**. Recording
the parameterisation here so the design is reviewed once.

From the architecture plan §3.2.2, §4.2 and §5 Phase 2:

- `A(t) = A₀ · exp(-λ · Δt)`; in-repo `exp(-0.015 · days)`.
- Basis changes from `created_at` to **`max(created_at, last_accessed_at)`** —
  items that continue to be accessed stay fresh.
- Composition is **multiplicative and applied after fusion**, as a modulation on
  the fused score: `combined = fused_score · activation_decay(...)`. It is *not*
  a third ranked list inside the fusion. Plan §4.2 is explicit about this
  ordering and I am not departing from it.
- Stale threshold `activation_score < 0.3`.
- Additive only: `activation_score FLOAT NOT NULL DEFAULT 1.0`, backfilled from
  `exp(-0.015 · days_since_created)` (plan §6.1).

The decay function itself already exists and is unchanged:
`node_agent_learning_retrieval_effect/handlers/handler_agent_learning_retrieval.py:20-24`
implements `math.exp(-0.015 * days_old)` on `created_at`. The plan's description
of this as "a one-line change" is still accurate *for that handler*.

### The open item

**On this node's path, `last_accessed_at` is not reachable, so the plan's
formula cannot be implemented as written.**

- `HandlerMemoryRetrieval` returns `ModelSearchResult`, which wraps
  `ModelMemorySnapshot`.
- `ModelMemorySnapshot` (`omnibase_core.models.omnimemory.model_memory_snapshot`)
  carries `created_at` and **no `last_accessed_at`**. It is
  `frozen=True, extra="forbid"`, and it lives in **omnibase_core** — a different
  published package, not this repo.
- `last_accessed_at` *does* exist in omnimemory, but on different models that are
  not on this path: `models/memory/model_memory_item.py:107` and
  `models/foundation/model_memory_data.py:145`.
- `activation_score` does not exist anywhere in `src/` or in the installed
  `omnibase_core`.

So there are three possible resolutions, and choosing between them is above what
I should decide alone:

1. **Decay on `created_at` only for this node**, and say so — a narrower feature
   than the plan describes, but self-contained in omnimemory.
2. **Plumb `last_accessed_at` onto the retrieval path** — requires adding a field
   to a frozen model in omnibase_core, i.e. a cross-repo change with its own
   ticket and release.
3. **Scope decay to the paths that already carry the field**
   (`ModelMemoryItem` / `ModelMemoryData`) and leave the snapshot path on
   `created_at`.

**This does not block hybrid search.** §4 step 4 already sequences decay second,
so PR 1 proceeds regardless. Flagging now rather than discovering it mid-PR.

---

## 7. Plan of record

| Step | Deliverable | State |
|---|---|---|
| 1 | This note | drafted, awaiting review |
| 2 | Fixture corpus + failing NDCG@10 test | **done — RED confirmed** |
| 3 | `search_hybrid` + `handler_fusion.py` | **done — GREEN** |
| 4 | Activation decay | separate PR; blocked on the §6 decision |

### Measured result

Mean NDCG@10 over the 24-query corpus. The vector-only column is the ranking
today's `search` operation would return.

| Family | vector-only | fused | delta |
|---|---|---|---|
| `exact_token` | 0.5661 | **0.9186** | **+0.3525** |
| `paraphrase` | 0.9777 | 0.9777 | +0.0000 |
| `agreement` | 0.9934 | 1.0000 | +0.0066 |
| overall | 0.8457 | **0.9654** | **+0.1197** |

### The finding: the plan's algorithm, applied literally, regresses

Plan §3.1.3 specifies RRF over both legs' returned results with no relevance
floor. Measured that way:

| Family | vector-only | fused (ungated) | delta |
|---|---|---|---|
| `exact_token` | 0.5661 | 0.8942 | +0.3281 |
| `paraphrase` | 0.9777 | 0.8446 | **−0.1331** |
| `agreement` | 0.9934 | 1.0000 | +0.0066 |

**Unweighted RRF costs 13.3 NDCG points on paraphrase queries.** The cause is
structural, not a corpus artefact: RRF awards whatever sits at a leg's rank 1
the full `1/(k+1)` weight regardless of that document's quality. On a paraphrase
query the lexical leg matches a stray lexeme and returns something merely
present rather than relevant — and that document then ties with, or outranks,
the correct answer the semantic leg had at rank 1.

Filtering each leg against **its own** floor before fusing recovers the whole
regression and improves `exact_token` a further 2.4 points. The floors are
per-leg and never compared across legs, so this does not reintroduce the
scale-incomparability problem RRF exists to avoid (§4).

Both numbers are asserted in the test suite rather than recorded here only:
`test_ungated_rrf_regresses_paraphrase_queries` fails if unweighted RRF ever
stops regressing, which is the signal to reconsider the gating.

### Gate results

* `mypy --strict` — clean, 311 source files, and clean on the test module
  itself.
* `pre-commit run --all-files` — 42 hooks, 0 failures.
* Full suite — 2625 passed, 200 skipped (skips are pre-existing, Memgraph
  unavailable locally).
* Existing retrieval callers unchanged: `search`, `search_text` and
  `search_graph` take the same paths they did before.
* Verified against `omnibase-core` 0.47.0 after OMN-16950 widened the cap.

One known-flaky exclusion, stated rather than quietly passed over:
`test_cosine_batch_performance` measures 107.9 ms against a 100 ms threshold on
this machine and fails identically on clean `dev`, so it is unrelated to this
change. Worth knowing that `ci.yml:410` sets `PERF_THRESHOLD_MS: "100"`, which
overrides the test module's own CI default of 1000 ms — a tolerance the module
comments were written to provide *because* "CI runners are significantly slower
than local machines and have unpredictable variance".

**Step 2 is complete and red.** `tests/unit/nodes/test_hybrid_retrieval_fusion.py`
against `tests/unit/nodes/fixtures/omn16765/hybrid_retrieval_corpus.json`: **8 passed,
5 failed**, every failure `ModuleNotFoundError: handler_fusion`.

The split is deliberate. The five NDCG self-tests and three corpus guards pass
**now**, before any implementation exists — so they are guards on the metric and
the corpus rather than on the feature, and a later failure in the fusion
assertions cannot be mistaken for a broken metric. The five fusion tests fail
only because the module is absent.

One assertion is expected to be genuinely hard, and that is the point:
`test_fusion_does_not_regress_paraphrase_queries`. Naive RRF gives a leg's
rank-1 result its full `1/(k+1)` weight regardless of whether that result is any
good, so on paraphrase queries — where the lexical leg returns noise — an
irrelevant document can tie or displace the correct semantic answer. If the
implementation cannot clear that assertion, the finding is a real limitation of
unweighted RRF on this input, not a corpus artefact, and it gets reported rather
than tuned away.

Definition of done for steps 2–3 is §5 of the ticket, unchanged: metric asserted
in the test, not eyeballed; existing retrieval callers unchanged; no handler on
the wrong side of the migration boundary; this note committed alongside the code.

---

## 8. Open questions for review

1. **§5 marker** — confirming `@pytest.mark.unit` over the ticket's
   "integration" wording, on the CI evidence above.
2. **§6 decay basis** — which of the three resolutions for the missing
   `last_accessed_at`. Needed before the decay PR, not before hybrid.
3. **§4 `k = 60`** — accepting the plan's default with the reasoning stated.
   Flagging that I accepted rather than tuned, since §4 of the ticket invited an
   alternative.
