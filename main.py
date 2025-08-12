import argparse

from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTSArgs, MCTS
from src.model.model_v1 import PenteNet
from torch import optim
import logging
from logging.handlers import RotatingFileHandler
import torch
import numpy as np

from src.train.arena import Arena
from src.train.nnet_player import NNetPlayer
from src.train.self_play import SelfPlayTrainer, SelfPlayTrainerArgs

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    level = logging.INFO

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--infer", action="store_true", help="Enable inference mode")
    parser.add_argument("--model-dir", type=str, help="Directory for model checkpoints")
    parser.add_argument("--model", type=str, help="Path of a model to load")
    parser.add_argument("--infer-mcts", action="store_true", help="Enable MCTS during inference")
    program_args = parser.parse_args()

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

    professional_games_training_iterations = 30
    board_size = 9 # Pente is usually played on 19x19
    if professional_games_training_iterations:
        # Professional game dataset is played on 19x19, so we force the board size
        board_size = 19
    player_count = 2 # Pente can be played with up to 4 players
    num_simulations = 15
    args = SelfPlayTrainerArgs(
        start_iteration=0,
        professional_games_training_iterations=professional_games_training_iterations,
        self_play_training_iterations=150,
        temp_threshold=int(num_simulations * 0.8),
        mcts_args=MCTSArgs(
            num_simulations=num_simulations,
        ),
        professional_games_training_examples_filepath='data/pente_dataset.txt',
        eval_iteration_interval=1,
        arena_num_games=35,
        batch_size=2048,
        batch_games=64,
        update_threshold=0.6,
        checkpoint_dir=program_args.model_dir if program_args.model_dir else f"pente-model-v1.3",
        should_checkpoint=True,
        max_training_examples = 0,
        board_size=board_size,
        player_count=2,
        debug=program_args.debug,
        should_use_arena=True
    )

    logger.info(f"Using checkpoint dir: '{args.checkpoint_dir}'")

    game = PenteGame(board_size=board_size, player_count=player_count)
    logger.info(f"Initialized pente game with board_size={board_size} and player_count={player_count}")
    pente_network = PenteNet(
        device,
        board_size=board_size,
        action_size=game.get_action_size(),
        hidden_fc_size=1,
    )
    optimizer = optim.Adam(pente_network.parameters(), lr=1e-3, weight_decay=1e-4)

    if program_args.model:
        logger.info(f"Trying to load model: '{program_args.model}'")
        start_iteration = PenteNet.load_checkpoint(program_args.model, pente_network, optimizer=optimizer)
        args.start_iteration = start_iteration

    # Compile model and do other torch initialization
    # TODO: this only works on Linux, and my hardware...
    if torch.cuda.is_available():
        logger.info("Compiling PenteNet with torch.compile()")
        pente_network.compile(fullgraph=True)

        torch.set_float32_matmul_precision('high')

    logger.info(
        f"Created PenteNet with {pente_network.get_parameter_count()} trainable parameters"
    )

    game = PenteGame(board_size=board_size, player_count=player_count)

    train = not program_args.infer
    if train:
        self_player_trainer = SelfPlayTrainer(
            game=game,
            net=pente_network,
            optimizer=optimizer,
            device=device,
            args=args,
        )
        self_player_trainer.train()
    else:
        model_checkpoint_dir = args.checkpoint_dir
        if program_args.model_name:
            model_name = program_args.model_name
        else:
            model_name = pente_network.get_checkpoint_file_name(args.professional_games_training_iterations)

        logger.info(f"Starting inference for model '{model_name}'")

        PenteNet.load_checkpoint(model_checkpoint_dir, pente_network, model_name, None)
        pente_network.compile(fullgraph=True)
        pente_network.eval()

        mcts1 = None
        mcts2 = None
        if program_args.infer_mcts:
            mcts1 = MCTS(game, pente_network, args.mcts_args)
            mcts2 = MCTS(game, pente_network, args.mcts_args)
        
        # TODO: move to static method on board
        def pretty_print_board(board: np.ndarray):
            """
            Prints a Pente board in a human-readable format with coordinate labels.

            Args:
                board: A 2D NumPy array representing the board state.
                       Assumes 1 for Player 1, -1 for Player 2, and 0 for empty.
            """
            if not isinstance(board, np.ndarray) or board.ndim != 2:
                print("Error: Input must be a 2D NumPy array.")
                return

            board_size = board.shape[0]

            p1_char = 'X'
            p2_char = 'O'
            empty_char = '.'

            COLS = "ABCDEFGHJKLMNOPQRST"
            if board_size > len(COLS):
                COLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            row_label_padding = len(str(board_size))
            header_prefix = ' ' * (row_label_padding + 1)
            header = header_prefix + " ".join(COLS[:board_size])
            print(header)

            for i in range(board_size):
                row_label = f"{i + 1:>{row_label_padding}} "

                row_items = []
                for j in range(board_size):
                    cell_value = board[i, j]
                    if cell_value == 1:
                        row_items.append(p1_char)
                    elif cell_value == -1:
                        row_items.append(p2_char)
                    else:
                        row_items.append(empty_char)

                print(row_label + " ".join(row_items))

            print(header)

        arena = Arena(
            player1=NNetPlayer(pente_network, mcts1, name="Player1"),
            player2=NNetPlayer(pente_network, mcts2, name="Player2"),
            game=game,
            debug=True,
            display=pretty_print_board
        )

        while True:
            arena.play_game()

            if input("Play another game? (y/n): ") != "y":
                break
