from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTSArgs
from src.model.model_v1 import PenteNet
from torch import optim
import logging
from logging.handlers import RotatingFileHandler
import torch

from src.train.self_play import SelfPlayTrainer, SelfPlayTrainerArgs
from src.train_v2 import train_v2, TrainArgs, SelfPlayArgs

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    level = logging.INFO

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    professional_games_training_iterations = 0
    board_size = 9 # Pente is usually played on 19x19
    if professional_games_training_iterations:
        # Professional game dataset is played on 19x19, so we force the board size
        board_size = 19
    player_count = 2 # Pente can be played with up to 4 players
    num_simulations = 9
    batch_size = 65536
    args = SelfPlayTrainerArgs(
        professional_games_training_iterations=professional_games_training_iterations,
        self_play_training_iterations=150,
        temp_threshold=int(num_simulations * 0.8),
        mcts_args=MCTSArgs(
            num_simulations=num_simulations,
        ),
        professional_games_training_examples_filepath='data/pente_dataset.txt',
        eval_iteration_interval=1,
        arena_num_games=35,
        batch_size=4096,
        batch_games=1,
        update_threshold=0.6,
        checkpoint_dir=f"pente-model-v1.2_{board_size}_{board_size}_{player_count}",
        should_checkpoint=False,
        max_training_examples = 0,
        board_size=board_size,
        player_count=2,
        debug=False,
        should_use_arena=False
    )

    game = PenteGame(board_size=board_size, player_count=player_count)
    logger.info(f"Initialized pente game with board_size={board_size} and player_count={player_count}")
    pente_network = PenteNet(
        device,
        board_size=board_size,
        action_size=game.get_action_size(),
        num_hidden_fc_value_layers=1024,
        hidden_fc_size=1,
    )
    optimizer = optim.Adam(pente_network.parameters(), lr=1e-3, weight_decay=1e-4)
    logger.info(
        f"Created PenteNet with {pente_network.get_parameter_count()} trainable parameters"
    )

    game = PenteGame(board_size=board_size, player_count=player_count)
    self_player_trainer = SelfPlayTrainer(
        game=game,
        net=pente_network,
        optimizer=optimizer,
        device=device,
        args=args,
    )
    self_player_trainer.train()