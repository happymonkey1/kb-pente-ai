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
from src.train.random_player import RandomPlayer
from src.train.self_play import SelfPlayTrainer, SelfPlayTrainerArgs

logger = logging.getLogger(__name__)

def clamped_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} is not a floating-point literal")

    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    return x

if __name__ == "__main__":
    level = logging.INFO

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--infer", action="store_true", help="Enable inference mode")
    parser.add_argument("--gpu", action="store_true", help="Attempt to run model on the first available GPU")
    parser.add_argument("--infer-games", type=int, help="Number of games to play in inference mode", default=1)
    parser.add_argument("--no-interactive", action="store_true", help="Disable any and all user interactions")
    parser.add_argument("--model-dir", type=str, help="Directory for model checkpoints")
    parser.add_argument("--model", type=str, help="Path of a model to load")
    parser.add_argument("--infer-mcts", action="store_true", help="Enable MCTS during inference")
    parser.add_argument("--mcts-sim", type=int, help="Number of MCTS simulations", default=15)
    parser.add_argument("--temp-threshold", type=int, help="Threshold for random move exploration during MCTS", default=15)
    parser.add_argument("--batch-games", type=int, help="Number of self-play games to execute per iteration", default=64)
    parser.add_argument("--batch-size", type=int, help="Size of training example batch(es)", default=32)
    parser.add_argument("--arena", action="store_true", help="Enable Arena during self-play training")
    parser.add_argument("--num-arena-games", type=int, help="Number of games to play during Arena evaluation", default=35)
    parser.add_argument("--arena-threshold", type=clamped_float, metavar="[0.0-1.0]", help="Threshold ([0.0, 1.0]) for whether the current model should promoted during Arena self-play.", default=0.6)
    parser.add_argument("--board-size", type=int, help="Size of the board", default=19)

    parser.add_argument("--raw-dataset", type=str, help="Path to the raw 'pro' examples dataset", default="data/pente_dataset.txt")
    parser.add_argument("--processed-dataset", type=str, help="Path to the 'pro' processed examples dataset", default="data/pente-dataset-processed.pkl")
    parser.add_argument("--force-dataset-processing", action="store_true", help="Force processing of raw 'pro' examples dataset")
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

    device = torch.device('cuda' if torch.cuda.is_available() and program_args.gpu else 'cpu')

    professional_games_training_iterations = 0
    board_size = program_args.board_size # Pente is usually played on 19x19
    if professional_games_training_iterations:
        # Professional game dataset is played on 19x19, so we force the board size
        board_size = 19
    player_count = 2 # Pente can be played with up to 4 players
    args = SelfPlayTrainerArgs(
        start_iteration=0,
        professional_games_training_iterations=professional_games_training_iterations,
        self_play_training_iterations=150,
        temp_threshold=program_args.temp_threshold,
        mcts_args=MCTSArgs(
            num_simulations=program_args.mcts_sim,
        ),
        watch_training_raw_dataset_filepath=program_args.raw_dataset,
        watch_training_processed_dataset_filepath=program_args.processed_dataset,
        force_watch_training_raw_dataset_processing=program_args.force_dataset_processing,
        eval_iteration_interval=1,
        num_arena_games=program_args.num_arena_games,
        batch_size=program_args.batch_size,
        batch_games=program_args.batch_games,
        update_threshold=program_args.arena_threshold,
        checkpoint_dir=program_args.model_dir if program_args.model_dir else f"pente-model-v1.3",
        should_checkpoint=True,
        max_training_examples = 0,
        board_size=board_size,
        player_count=2,
        debug=program_args.debug,
        should_use_arena=program_args.arena
    )

    # Print arguments
    logger.info(f"Arguments:")
    logger.info(f"  model: '{program_args.model}'")
    logger.info(f"  checkpoint_dir: '{args.checkpoint_dir}'")
    logger.info(f"  num_simulations: {args.mcts_args.num_simulations}")
    logger.info(f"  batch_games: {args.batch_games}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  board_size: {args.board_size}")
    logger.info(f"  device: {device}")
    logger.info(f"  ----------------------")
    logger.info(f"  arena: {args.should_use_arena}")
    logger.info(f"  num_arena_games: {args.num_arena_games}")
    logger.info(f"  arena_update_threshold: {args.update_threshold}")
    logger.info(f"  ----------------------")
    logger.info(f"  raw_dataset: {args.watch_training_raw_dataset_filepath}")
    logger.info(f"  processed_dataset: {args.watch_training_processed_dataset_filepath}")
    logger.info(f"  force_dataset_processing: {args.force_watch_training_raw_dataset_processing}")
    logger.info(f"  ----------------------")
    logger.info(f"  infer: {program_args.infer}")
    logger.info(f"  infer_mcts: {program_args.infer_mcts}")
    logger.info(f"  infer_games: {program_args.infer_games}")
    logger.info(f"  no_interactive: {program_args.no_interactive}")

    game = PenteGame(board_size=board_size, player_count=player_count)
    logger.info(f"Initialized pente game with board_size={board_size} and player_count={player_count}")
    pente_network = PenteNet(
        device,
        board_size=board_size,
        action_size=game.get_action_size(),
        num_res_blocks=8,
        num_channels=512,
        hidden_fc_size=768,
    )

    optimizer = optim.AdamW(pente_network.parameters(), lr=1e-3, weight_decay=1e-4, foreach=True)

    # Compile model and do other torch initialization
    # TODO: this only works on Linux, and my hardware...
    if torch.cuda.is_available():
        logger.info("Compiling PenteNet with torch.compile()")
        pente_network.compile(fullgraph=True)

        torch.set_float32_matmul_precision('high')

    if program_args.model:
        logger.info(f"Trying to load model: '{program_args.model}'")
        start_iteration = PenteNet.load_checkpoint_from_path(program_args.model, pente_network, optimizer=optimizer)
        args.start_iteration = start_iteration


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
        logger.info(f"Starting inference")

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

        p1_wins, p2_wins, draws = 0, 0, 0
        for i in range(program_args.infer_games):
            stats = arena.play_game()
            if stats.winner == 1:
                p1_wins += 1
            elif stats.winner == -1:
                p2_wins += 1
            else:
                draws += 1

            if not program_args.no_interactive and input("Play another game? (y/n): ") != "y":
                break

        logger.info(f"p1 wins: {p1_wins}")
        logger.info(f"p2 wins: {p2_wins}")
        logger.info(f"draws: {draws}")
