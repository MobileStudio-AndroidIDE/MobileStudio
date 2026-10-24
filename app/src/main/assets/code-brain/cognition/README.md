# RENA-BR Cognitive Growth Subsystem

`cognition/` is an additive, model-free growth subsystem. It does **not**
use pretrained weights, an external LLM, or a remote AI API. Learning is
counting/Hebbian VSA association plus verification-gated skill reuse.

## What actually learns

```
actual task
   |
   v
failure / surprise ---- goal (persisted) ---- practice on fresh instances
   ^                                               |
   |                                               v
skill registry <---- verification gate <---- experiment + holdout + sandbox
   |
   +---- reuse on novel task ---- transfer evidence / reliability numbers
```

A skill is a *candidate* until it has at least three successful verified
attempts, a success rate of at least 0.75, and at least one held-out
success (defaults are configurable). Verification is domain-specific and
independent of the candidate's own claim:

- program synthesis: held-out outputs + generated Python executed in a sandbox;
- sequences: held-out continuation + strategy recomputation;
- logic: truth table + checked resolution proof/model;
- graphs/data: sandboxed code + separate reference computation;
- images: held-out pixels re-extracted from the decoded artifact;
- files: actual filesystem state, with denied-path checks;
- grounded conversation: answer plus supported evidence triples.

Only active skills are exposed through the bridge. Retired or tampered
skills are deactivated, not silently retained.

## Modules

| Module | Role |
|---|---|
| `vsa.py` | 2048-d bipolar MAP-VSA: bind/unbind/bundle/permute, text/sequence/feature encoders, cleanup, capacity probe |
| `memory.py` | concept/relation memory, bounded working memory, prioritized episodic replay, consolidation |
| `predictive.py` | n-gram + VSA association prediction, surprise and bounded planning |
| `compositional.py` | role/filler queries, analogy, transitive closure, rule induction |
| `dsl.py` | typed expression DSL, enumerative synthesis with holdout checks, persistent macros |
| `domains.py` | math/code, sequence, logic, graph, image, file, conversation and tabular-data adapters |
| `skills.py` | verification gate, skill lifecycle, integrity checksum, promotion/retirement ledger, transfer evidence |
| `sandbox.py` | static permission checks, separate-process execution, manifests, guarded self-modification with rollback |
| `tools.py` | verified web/video/image/data tool probes and gated callable wrappers |
| `growth.py` | bounded failure -> goal -> practice -> verify -> skill -> transfer engine |
| `bridge.py` | optional synchronization with `brain.capability_registry.CapabilityRegistry` |
| `journal.py` | bounded append-only growth journal and limitations record |
| `semantic.py` | model-free multimodal encoders: typo/emoji text, image patches/histograms, video frames, AST code, math trees, role/filler graphs |
| `concept_space.py` | prototype/exemplar concepts, similarity classification, merging, clustering, persistence |
| `active.py` | discrete active inference: beliefs, likelihood, surprise, variational free energy, action ranking |
| `metacognition.py` | success/confidence calibration (Brier/ECE), error taxonomy, attempt decisions, event budgets, reflection |
| `curiosity.py` | expected-learning-progress scoring, interest memory, bounded micro-task proposals |
| `learning.py` | replay buffer, contrastive VSA association, UCB strategy selection, developmental stages, consolidation |
| `math_engine.py` | typed expression parser, evaluator, symbolic differentiation, simplification, linear solver |
| `actuators.py` | evidence-returning code/math/image/chart/video/audio/web/file/data actuators and registry |
| `rena.py` | `RenaBrain`: unified perception, concepts, active inference, metacognition, learning and action |
| `autonomous.py` | bounded autonomous loop combining growth cycles, curiosity probes, consolidation, augmentation and primitive invention |
| `augmentation.py` | self-augmentation: recombine typed primitives/macros, generate fresh-task verifiers, gate and roll back new skills |
| `invention.py` | primitive invention: abstract numeric constants/subexpressions from verified programs into new parametric typed primitives |
| `sensors.py` | virtual sensor channels, measurable channel composition, persistence and verification |

## Usage

```bash
# one bounded growth run over all domains
python3 cli.py grow --cycles 6 --time-budget 120

# grow only one domain
python3 cli.py cognition grow --domain math --cycles 3

# inspect persisted state
python3 cli.py cognition status
python3 cli.py cognition capabilities

# verify tools first; web is only probed with --network
python3 cli.py cognition tools --network

# last report
python3 cli.py cognition report --json

# unified brain: status, perception, retrieval, math and action
python3 cli.py cognition brain
python3 cli.py cognition observe --modality text --text "python code"
python3 cli.py cognition think "python code" --modality text
python3 cli.py cognition math "x^2+2*x" --operation differentiate
python3 cli.py cognition act math --expression "2*x=10" --operation solve --force
python3 cli.py cognition act image --prompt "a bird" --operation generate
python3 cli.py cognition act image --request \
  '{"operation":"chart","labels":["A","B"],"values":[3,7]}'
python3 cli.py cognition act audio --request \
  '{"operation":"speak","text":"hello"}'
python3 cli.py cognition act web --request "Photosynthesis" --network
python3 cli.py cognition curriculum --count 6
python3 cli.py cognition augment --cycles 2 --candidates 4
python3 cli.py cognition invent --kind primitive --candidates 6
python3 cli.py cognition invent --kind sensor --modality text \
  --sample "alpha 1" --sample "beta 22" --sample "gamma 333"
python3 cli.py cognition sensor list

# bounded autonomous development (growth + probes + consolidation + augmentation + primitive invention)
python3 cli.py cognition autonomous --cycles 4 --time-budget 120 --json
```

Python integration:

```python
from cognition import GrowthConfig
from cognition.bridge import CognitionRuntime
from brain.code_brain import CodeBrain

brain = CodeBrain()
runtime = CognitionRuntime(
    GrowthConfig(max_cycles=8, time_budget_seconds=120),
    brain=brain,
)
report = runtime.grow()
runtime.verify_tools(include_network=True)
runtime.sync_capabilities()   # verified skills/tools -> brain.capability_registry
runtime.close()
```

Unified (perception/action) integration:

```python
from cognition import GrowthConfig, RenaBrain

brain = RenaBrain(GrowthConfig(dimension=2048))
brain.observe("text", "python code function", label="code")
print(brain.think("python code"))
print(brain.act("math", {"operation": "solve",
                         "expression": "2*x=10"}, force=True))
print(brain.augment(cycles=2, candidates=4))
print(brain.invent_primitives(cycles=1, candidates=6))
sensor = brain.invent_sensor("text", ["a 1", "b 22", "c 333"])
print(brain.read_sensor(sensor["sensor"], "d 4444"))
brain.close()
```

The existing `CodeBrain` pipeline is unchanged unless the bridge is
explicitly constructed. `runtime.sync_capabilities()` adds only verified
capabilities and deactivates stale `cognition.*` entries. `RenaBrain` is
also additive: it does not route or replace the existing CodeBrain
pipeline.

## Data locations

Default: `~/.rena_cognition` (override with `RENA_COGNITION_DIR` or
`GrowthConfig(data_dir=...)`).

- `skills/skills.json` — skills, promotion/retirement ledger, checksums;
- `library/program_library.json` — verified macros;
- `concepts/`, `episodes/`, `predictive/` — durable memory;
- `goals.json` — open/abandoned growth goals;
- `journal/` — growth events and limitations;
- `tools.json` — tool verification history;
- `reports/` — last and timestamped growth/autonomous reports;
- `augmentation.json` — self-augmentation sessions, candidates, verifier
  evidence and adoption/rollback status;
- `sensor_library.json` — invented virtual sensors and their channel
  composition;
- `invention.json` — primitive/sensor invention sessions and evidence;
- `workspace/`, `artifacts/` — file-domain and image/video/audio outputs;
- `concept_space.npz` — prototypes, counts, modalities and spread;
- `metacognition.json`, `curiosity.json`, `learning.json` — calibration,
  interests and developmental state;
- `learning_state.npz` — persisted contrastive associations/targets;
- `concepts/`, `episodes/`, `predictive/` are shared by the unified brain
  and the growth engine.

## Measured results in this environment

Termux/Android, Python 3.14.8, NumPy 2.5.4, no network for default runs:

- `benchmark/cognition_benchmark.py`, all 9 domains, 3 cycles each,
  dimension 2048: **27/27** synthetic tasks verified, **20** active
  skills, **20/20** transfer checks passed, 29.471 s.
- A 12-cycle all-domain growth run: **12/12** tasks verified, **11**
  skills promoted, transfer **10/11**, 16.0 s.
- VSA capacity probe with 40 superposed pairs, 25% symbol noise,
  dimension 2048, seed 13: **62.5%** exact cleanup.
- Tool probes: video generation+decode+analysis, image render+decode+
  region measurement, and tabular aggregates passed. With `--network`,
  two Wikipedia summary retrievals passed; DuckDuckGo returned no
  results in this environment and is only recorded as informational.
- `tests/test_cognition.py`: **33/33** tests passed in 31.8 s (VSA,
  memory, synthesis, abstraction, semantic encoders, concepts, active
  inference, math, calibration, curiosity, learning, actuators,
  `RenaBrain` persistence, tools, self-augmentation, learned-macro reuse,
  primitive invention and virtual-sensor synthesis).
- `tests.test_all`: 260 tests ran; exactly 7 pre-existing environment
  errors remain (`/root` is read-only on Android; `_NeuralCoreUnavailable`
  has no `think`). No new failures.
- Composition transfer benchmark: source solved cold at 12,064
  candidates; mined macro `m_2cb157ce4d`; target failed cold at 30,000
  candidates but solved warm at 12,359; held-out check passed in
  6.5398 s.
- Self-augmentation benchmark, 2 cycles / 4 candidates per cycle, 3
  fresh verification trials each: **8/8** candidates verified and
  adopted, macros 0 -> 8, active skills 0 -> 8, 5.779 s; 4/4
  second-cycle macro-based candidates were adopted, demonstrating that
  adopted macros become primitives for later augmentation. The verifier
  is generated from each candidate's own composition plus a held-out
  split and sandboxed code execution; this is not external semantic proof.
- Invention benchmark: **3/3** parametric primitives adopted
  (0 -> 3, 0.640 s) by abstracting numeric constants/subexpressions from
  seeded verified expressions; **1/12** virtual sensor candidates adopted
  (0.143 s) by composing text channels. The sensor check verifies
  determinism, finiteness and dispersion on supplied samples.
- Autonomous smoke run, 2 cycles / 40 s budget: 2/2 cycles, growth 3/4
  verified, curiosity probes 1/2, self-augmentation 6/6 adopted, 4
  parametric primitives invented from 4 candidates via the foundation
  seed path, 1 consolidation, no budget violation. This is a smoke test,
  not a capability claim.

These are numbers on **generated, internally verified tasks**, not an
external benchmark and not evidence of general intelligence.

## Reproduce

```bash
python3 -m unittest tests.test_cognition -v
python3 benchmark/cognition_benchmark.py --data-dir "$(mktemp -d)"
python3 benchmark/augmentation_benchmark.py
python3 benchmark/invention_benchmark.py
python3 benchmark/composition_benchmark.py
```

## Safety and limits

- Every loop has cycle, attempt and wall-clock budgets; exhaustion is
  recorded (`time_budget_exhausted`, etc.), never retried forever.
- The sandbox is static AST filtering + a separate process + file
  permissions + CPU/file-descriptor limits. It is not a kernel security
  boundary. On Android/Bionic, `RLIMIT_AS` is incompatible with the
  linker and is skipped; CPU/file limits remain.
- Self-modification is **off** by default. When enabled, a patch must
  match existing text, parse, pass permission checks, run a supplied
  test command, and is restored byte-for-byte on any failure.
- Synthetic tasks only; no claim of open-ended natural-language
  understanding or natural-world visual recognition.
- The `readThis.txt` target mentions a 100k-dimensional unified space.
  The implemented default is 2048 (configurable); no 100k capacity or
  equivalence claim is made.
- Text/image/video/code encoders are hand-designed VSA features, not
  learned perception. Image/video actuators use the repository's
  procedural scene generators and therefore only support their known
  objects/scenes.
- Binding is the real-valued MAP elementwise product, not the FFT
  circular convolution sketched in `readThis.txt`; both are VSA binding
  schemes, but the implementation does not claim FFT equivalence.
- "Active inference" is a bounded discrete belief update with explicit
  surprise/free-energy numbers, not a full hierarchical generative model.
  There is no separate intent-prototype table or multi-step mental
  world-model rollout; concepts/prediction and real bounded probes cover
  those roles.
- The math engine is a small algebra system (no general CAS), and the
  autonomous loop's non-math probes defer to growth-domain verification.
- Self-augmentation composes existing typed operators/macros. Primitive
  invention abstracts a numeric constant or subexpression from verified
  programs into a new parametric primitive; when no learned numeric
  expression is available, three foundation patterns (sum/2, len>3,
  len*2) seed the search. It does not invent an arbitrary base operator
  or arbitrary code. Virtual sensors compose
  existing measurement channels; a physical sensor still requires a
  transducer and driver. Generated verifiers and sensor checks validate
  compositional behaviour, code execution and measured dispersion, not
  external real-world semantics. Adoption still uses the normal
  three-success verification gate; failed adoption is retired and its
  new macro is removed.
- Web retrieval is limited to Wikipedia summaries; general search may be
  rate-limited. Tool capabilities are not registered unless the probe
  passes.
