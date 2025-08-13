

## Usage

Run training starting from scratch:
```bash
```

Run training starting from a checkpoint:
```bash
uv run python main.py --model-dir=pente-model-v1.4 --model=pente-model-v1.4/checkpoint-32_19_361_5_128_1.pth.tar --batch-games=1 --arena
```

Run training starting from scratch and force processing of a raw dataset:
```bash
uv run python main.py --model-dir=pente-model-v1.5 --batch-games=1 --arena --raw-dataset=data/pente_dataset.txt --processed-dataset=data/pente-dataset-processed.pkl --force-dataset-processing
```