
# kb-pente-ai

AlphaZero inspired machine learning applied to the board game Pente.

Spiritual successor to my [college thesis' source code](https://github.com/happymonkey1/uci-chc-NNUE-thesis).

## Dependencies
- [uv](https://docs.astral.sh/uv/)
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Developer Setup

- `uv sync`

## Usage

Run training starting from scratch:
```bash
uv run python main.py --model-dir=pente-model-v1.9 --batch-games=64 --batch-size=512 --arena --num-arena-games=35 --temp-threshold=5 --mcts-sim=40 --gpu
```

Run training starting from a checkpoint:
```bash
uv run python main.py --model-dir=pente-model-v1.6 --model=pente-model-v1.5/checkpoint-30_19_361_5_128_1.pth.tar --batch-games=96 --batch-size=1024 --arena --num-arena-games=35 --temp-threshold=9 --mcts-sim=15 --gpu
```

Run training starting from scratch and force processing of a raw dataset:
```bash
uv run python main.py --model-dir=pente-model-v1.5 --batch-games=1 --arena --raw-dataset=data/pente_dataset.txt --processed-dataset=data/pente-dataset-processed.pkl --force-dataset-processing
```

Start inference and self-play evaluation
```bash
uv run python main.py --model-dir=PATH_TO_MODEL_DIR --model=PATH_TO_MODEL --infer --infer-mcts --batch-games=64 --batch-size=512 --arena --num-arena-games=35 --temp-threshold=5 --mcts-sim=40 --gpu
```