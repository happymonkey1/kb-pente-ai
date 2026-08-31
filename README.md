# kb-pente-ai

AlphaZero-inspired policy/value learning for Pente.

The current implementation is a correctness-first reconstruction of the original project. It uses complete Pente state, tested MCTS, batched self-play inference, versioned artifacts, and structured telemetry.

## Requirements

- Python 3.10 through 3.12
- [uv](https://docs.astral.sh/uv/)
- CUDA-capable PyTorch installation for GPU training

## Setup

~~~bash
uv sync
~~~

The repository scripts use the local .venv directly so Windows executables cannot be selected accidentally under WSL.

## Validation

Run the complete local gate:

~~~bash
./script/check.sh
~~~

Each command is mirrored to the terminal and preserved in a uniquely named temporary log.

The gate covers compilation, mypy, all unit and regression tests, batched-search equivalence, full state contracts, model lifecycle, professional preprocessing, replay persistence, telemetry, tactical fixtures, and the 32-example tiny-learning proof.

Run only the explicit tiny-learning proof:

~~~bash
./script/run-venv.sh python script/verify-tiny-learning.py
~~~

The evidence-oriented scripts return a nonzero status when their declared gate fails:

~~~bash
./script/run-venv.sh python script/verify-professional-learning.py METRICS.jsonl
./script/run-venv.sh python script/benchmark-batched-search.py
./script/run-venv.sh python script/verify-random-play.py CHECKPOINT
./script/run-venv.sh python script/verify-model-improvement.py CANDIDATE BASELINE
~~~

## Rulesets

standard is the default and matches the checked-in professional dataset: Player 1's first move is the center intersection. The rules follow the [Pente.org game rules](https://www.pente.org/help/helpWindow.jsp?file=playGameRules).

tournament also requires Player 1's second move to be at least three intersections from center.

freestyle permits any empty opening and is useful for small deterministic tests.

The ruleset is stored in processed data and checkpoint metadata. Artifacts from different rulesets cannot be mixed silently.

## Debug training

This small CPU run is useful for exercising self-play, batched MCTS, training, checkpointing, and telemetry:

~~~bash
./script/run-venv.sh python main.py \
  --board-size 5 \
  --ruleset freestyle \
  --self-play-iterations 1 \
  --batch-games 8 \
  --active-games 8 \
  --batch-size 32 \
  --mcts-sim 8 \
  --model-blocks 1 \
  --model-channels 16 \
  --model-hidden-size 32 \
  --telemetry-file metrics/debug.jsonl
~~~

## Professional data

The raw dataset is processed transactionally. Every accepted target is paired with its legal pre-move state. Games are assigned deterministically to training or validation before positions are aggregated, preventing game-level leakage. Identical complete states aggregate their observed policy and value targets within their split.

To rebuild the versioned cache and run one professional training iteration:

~~~bash
./script/run-venv.sh python main.py \
  --professional-iterations 1 \
  --self-play-iterations 0 \
  --force-dataset-processing \
  --processed-dataset data/pente-dataset-v3.pkl
~~~

The corrected four-plane input and target semantics are incompatible with legacy checkpoints and processed caches. The loader rejects them with an explicit error.

Professional validation is measured before and after the learner update. The verifier's defaults require a 5 percent cross-entropy reduction, top-1 and top-5 accuracy gains, and no material value-MSE regression.

## Replay and resume

Replay entries record whether they came from professional data or self-play. Once both sources exist, `--professional-replay-fraction` controls their learner-batch mix. Sampling uses replacement when necessary so `batch size * learner steps` remains the actual learner budget.

Replay snapshots are versioned and include the training run identifier, generation, ruleset, board size, position schema, and training-example schema. Persisted generations are immutable, while `replay-latest.pkl` is an atomically replaced alias. A resumed checkpoint selects its referenced replay generation even if an interrupted newer write advanced the alias. Resuming training never silently creates an empty replay.

Resume in the original model directory:

~~~bash
./script/run-venv.sh python main.py \
  --model MODEL_DIR/latest.pth.tar \
  --model-dir MODEL_DIR
~~~

When writing continued checkpoints elsewhere, identify the source replay explicitly:

~~~bash
./script/run-venv.sh python main.py \
  --model SOURCE/checkpoint-N.pth.tar \
  --resume-replay SOURCE/replay-latest.pkl \
  --model-dir DESTINATION
~~~

If a supervised checkpoint intentionally has no replay, `--seed-replay-from-professional` rebuilds the learner replay from the compatible professional cache. This is explicit because it changes training state.

## First serious 19 by 19 configuration

Use this only after the local gate and a small debug run pass on the target machine:

~~~bash
./script/run-venv.sh python main.py \
  --gpu \
  --compile \
  --ruleset standard \
  --board-size 19 \
  --model-blocks 6 \
  --model-channels 128 \
  --model-hidden-size 256 \
  --professional-iterations 1 \
  --self-play-iterations 5 \
  --mcts-sim 64 \
  --batch-games 512 \
  --active-games 128 \
  --batch-size 512 \
  --learner-steps 256 \
  --temp-threshold 16 \
  --max-training-examples 1000000 \
  --professional-replay-fraction 0.25 \
  --professional-value-loss-weight 0.25 \
  --self-play-value-loss-weight 1.0 \
  --replay-checkpoint-interval 1 \
  --minimum-batch-occupancy 0.80 \
  --minimum-mean-root-children 4 \
  --maximum-search-collapse-rate 0.25 \
  --maximum-invalid-policy-fallbacks 0 \
  --maximum-zero-visit-fallbacks 0 \
  --arena \
  --eval-interval 3 \
  --num-arena-games 40 \
  --arena-opening-plies 4 \
  --seed 103 \
  --model-dir pente-model-p7-e001-seed103 \
  --telemetry-file metrics/p7-e001-seed103.jsonl
~~~

This bounded command is the P7-E001 baseline, not an open-ended default. It performs one professional initialization and five self-play iterations. Completed games are replenished until all 512 games are generated, keeping approximately 128 active outside the final drain. Search-health thresholds abort before bad self-play enters replay. Arena openings are seeded and paired: each opening is replayed with model colors swapped. Arena results are measurements. They never reject the latest model or stop the learning stream.

Each training invocation writes `run-manifest-step-N.json` beside its checkpoints. The manifest records the run identifier, complete command and parsed configuration, Git state, a SHA-256 fingerprint of executable source files, artifact schemas, Python/Torch/CUDA identity, hardware, and output paths. Every telemetry record carries the same run identifier, and appending to a telemetry file owned by another run is rejected.

## Inference

~~~bash
./script/run-venv.sh python main.py \
  --infer \
  --model PATH_TO_SCHEMA_V2_CHECKPOINT \
  --infer-mcts \
  --infer-games 40
~~~

## Telemetry

Training writes JSONL records with a stable schema. Current metrics include:

- root children visited, visit entropy, and maximum visit share;
- unique trajectories and positions;
- leaf evaluations per second;
- min, mean, median, p95, and maximum inference batch size;
- sustained batch occupancy outside the final active-game drain;
- duplicate leaf rate and leaf-evaluation throughput;
- self-play positions and games per second;
- policy loss and KL;
- value MSE, absolute error, and bias;
- first-player, second-player, draw, and capture-win counts;
- replay size, source mix, uniqueness, generation lag, and age;
- arena game and paired-opening results against the prior model and random play;
- unique paired openings and model-identity wins by color;
- search-collapse warnings and invalid-policy fallbacks;
- fixed tactical-suite accuracy;
- sampled mean, p95, and maximum GPU utilization for self-play and learning;
- device-wide memory-controller activity plus peak Torch allocation and reservation.
- normalized process CPU utilization and resident memory for self-play and learning,
  including sample counts, mean/p95/maximum utilization, mean/peak RSS, logical core
  count, and sampling errors.

Pass `--replay-sample-file replays/training.jsonl` to emit at most eight
deterministically selected, validated completed-game samples after each successful
self-play iteration. These browser-safe records contain only the run and game
identifiers, board configuration, action sequence, winner, and win reason. They do
not serialize positions, policies, models, search trees, or replay-buffer entries.

Passing `--gpu` is strict. Training exits with a diagnostic if Torch cannot access CUDA; it never silently falls back to CPU. WSL GPU access may need to be granted outside a restricted execution sandbox even when `nvidia-smi` works in an ordinary WSL terminal.

## Monitoring dashboard

Start the local monitoring server from the repository root:

~~~bash
./script/run-venv.sh python -m src.monitoring \
  --metrics-root metrics \
  --replay-root replays
~~~

Open `http://127.0.0.1:8765`. The server watches JSONL telemetry beneath `metrics` and refreshes the dashboard every two seconds. It also finds run manifests in the current directory and immediate child directories for the NN architecture view. Add other model locations by repeating `--manifest-root PATH`.

Test launching is disabled by default. To enable the reviewed sample catalog:

~~~bash
./script/run-venv.sh python -m src.monitoring \
  --metrics-root metrics \
  --replay-root replays \
  --test-config docs/monitoring-tests.example.json \
  --test-root . \
  --test-log-root .monitoring/test-runs
~~~

The Replay tab reads safe JSONL samples beneath `--replay-root`. Training does not currently emit these browser samples automatically; replay-buffer `.pkl` checkpoints are not browser replay files.

See [`docs/monitoring.md`](docs/monitoring.md) for manifest fields, CPU and CUDA metrics, replay sample format, APIs, security limits, and test catalog configuration.

The implementation journal and original diagnosis are stored in the kb-pente-ai Codex vault.
