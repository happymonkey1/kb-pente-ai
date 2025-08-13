
# kb-pente-ai

AlphaZero inspired machine learning applied to the board game Pente.

Spiritual successor to my [college thesis' source code](https://github.com/happymonkey1/uci-chc-NNUE-thesis).



## Usage

Run training starting from scratch:
```bash
```

Run training starting from a checkpoint:
```bash
uv run python main.py --model-dir=pente-model-v1.6 --model=pente-model-v1.5/checkpoint-30_19_361_5_128_1.pth.tar --batch-games=96 --batch-size=1024 --arena --num-arena-games=35 --temp-threshold=9 --mcts-sim=15 --gpu
```

Run training starting from scratch and force processing of a raw dataset:
```bash
uv run python main.py --model-dir=pente-model-v1.5 --batch-games=1 --arena --raw-dataset=data/pente_dataset.txt --processed-dataset=data/pente-dataset-processed.pkl --force-dataset-processing
```