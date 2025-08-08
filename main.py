from src.model.model_v1 import PenteNet
from src.train import train
from torch import optim
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    level = logging.DEBUG

    file_handler = RotatingFileHandler(
        'kb-pente-ai.log',
        mode='a',
        maxBytes=5*1024*1024,
        backupCount=2,
        encoding=None,
        delay=False
    )
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('kb-penta-ai.log'),
            logging.StreamHandler()
        ]
    )
    logger.info("Starting training")

    pente_network = PenteNet()
    optimizer = optim.Adam(pente_network.parameters(), lr=1e4)

    train(
        num_iterations=10_000,
        batch_games=1,
        eval_interval=100,
        net=pente_network,
        optimizer=optimizer
    )