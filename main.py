from src.model.model_v1 import PenteNet
from src.train import train
from torch import optim

if __name__ == "__main__":
    pente_network = PenteNet()
    optimizer = optim.Adam(pente_network.parameters(), lr=1e4)

    train(
        num_iterations=10_000,
        batch_games=128,
        eval_interval=100,
        net=pente_network,
        optimizer=optimizer
    )