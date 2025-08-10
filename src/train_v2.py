
import logging
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from dataclasses import dataclass
import numpy as np

from src.game.game import Game
from src.game.pente.pente_board import PenteBoard
from src.pente_dataloader import PenteDataset
from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTS, MCTSArgs
from src.model.model_v1 import PenteNet
from src.train.player import Player
from src.train.random_player import RandomPlayer

logger = logging.getLogger(__name__)

@dataclass
class SelfPlayArgs:
    temp_threshold: int
    mcts_args: MCTSArgs

@dataclass
class TrainArgs:
    training_iterations: int
    batch_size: int
    batch_games: int
    eval_interval: int
    eval_iterations: int
    should_checkpoint: bool
    checkpoint_dir: str
    self_play_args: SelfPlayArgs
    player_count: int = 2
    board_size: int = 19
    debug: bool = False

@dataclass
class EvalData:
    score: float
    moves: float
    captures: float

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth.tar"):
    """Saves model and training parameters."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(net, optimizer, checkpoint_dir, filename="checkpoint.pth.tar"):
    """Loads model and training parameters."""
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(filepath):
        logger.warning(f"No checkpoint found at '{filepath}'. Starting from scratch.")
        return 0 # Return starting iteration 0

    # When loading a checkpoint, map storage to the same device the model is on
    # This prevents errors if you save on a GPU and load on a CPU, or vice-versa.
    map_location = next(net.parameters()).device
    checkpoint = torch.load(filepath, map_location=map_location)

    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_iteration = checkpoint['iteration']

    logger.info(f"Checkpoint loaded from '{filepath}'. Resuming from iteration {start_iteration}.")
    return start_iteration

def self_play(game: 'PenteGame', net: 'PenteNet', args: SelfPlayArgs) -> list[tuple[np.ndarray, np.ndarray, float]]:
    examples = []
    board = game.init_board()
    current_player = Game.PLAYER_ONE
    step = 0
    mcts = MCTS(game, net, args.mcts_args)

    while True:
        step += 1
        canonical_board = game.get_canonical_form(board, current_player)
        temp = int(step < args.temp_threshold)

        pi = mcts.get_action_prob(canonical_board, temp=temp)

        symmetries = game.get_symmetries(canonical_board, pi)
        for b, p in symmetries:
            examples.append([b, current_player, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, next_player = game.apply_action(pente_board=board, player=current_player, action=action)

        is_terminal, winner = game.check_game_end(board, next_player)
        if is_terminal:
            return [(x[0], x[2], winner * (-1 ** (x[1] != next_player))) for x in examples]

        current_player = next_player

def evaluate(
    net: 'PenteNet',
    player: 'Player',
    args: TrainArgs,
) -> EvalData:
    # TODO: switch between player 1 and 2
    pente_net_player = PenteGame.PLAYER_ONE
    game = PenteGame(args.board_size, args.player_count)
    board = PenteBoard.new_board(args.board_size, args.player_count)
    current_player = PenteGame.PLAYER_ONE

    moves = 0
    while True:
        move = None
        if current_player == PenteGame.PLAYER_ONE:
            legal_moves = game.get_valid_moves(board, current_player)
            p,v = net.predict(board)
            p = p.cpu().numpy() * legal_moves
            move = torch.argmax(torch.from_numpy(p)).item()
        else:
            move = player.play(board, current_player)
        moves += 1

        board, next_player = game.apply_action(board, current_player, move)
        terminal, winner = game.check_game_end(board, current_player)
        if terminal:
            if args.debug:
                logger.info(f"\n{board.board}")
                logger.info(f"Winner: {current_player}")
            break

        current_player = next_player

    return EvalData(
        score=1 if current_player == pente_net_player else 0,
        moves=moves,
        captures=int(board.captures[pente_net_player - 1]),
    )

def train_v2(
    net,
    optimizer,
    device,
    args: TrainArgs
):
    logger.info(f"Entered training loop on device: {device}")

    policy_loss = torch.nn.CrossEntropyLoss()
    value_loss = torch.nn.MSELoss()

    if args.debug:
        logger.debug("Debug mode enabled")
        torch.autograd.set_detect_anomaly(True)

    start_iteration = 0
    if args.should_checkpoint:
        start_iteration = load_checkpoint(net, optimizer, args.checkpoint_dir)

    for iteration in range(start_iteration, args.training_iterations):
        iter_start_time = time.time()
        logger.info(f"=== Iteration {iteration + 1}/{args.training_iterations} ===")

        net.eval()
        sp_start_time = time.time()
        game = PenteGame(board_size=args.board_size, player_count=args.player_count)
        training_examples = []
        for _ in range(args.batch_games):
            training_examples += self_play(game, net, args.self_play_args)
        sp_time = time.time() - sp_start_time
        logger.info(f"Generated {args.batch_games} training examples in {sp_time:.2f}s "
                    f"({args.batch_games / sp_time:.2f} examples/sec)")

        if not training_examples:
            logger.warning("No training examples were generated in this iteration. Skipping training.")
            continue

        flat_examples = [item for sublist in training_examples for item in sublist]
        logger.info(f"Collected {len(flat_examples)} training positions ({len(flat_examples) / sp_time} positions/sec)")

        net.train()
        train_start_time = time.time()
        total_loss, total_policy_loss, total_value_loss, num_batches = 0, 0, 0, 0

        dataloader = DataLoader(PenteDataset(training_examples), batch_size=args.batch_size, shuffle=True)
        for batch_idx, batch in enumerate(dataloader, 1):
            states, target_policies, target_values = batch
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device).view(-1, 1)

            optimizer.zero_grad()

            pred_policies_logits, pred_values = net(states)

            if args.debug:
                if torch.isnan(states).any():
                    logger.error("Detected NaN in states")

                if torch.isnan(target_policies).any():
                    logger.error("Detected NaN in target_policies")

                if torch.isnan(target_values).any():
                    logger.error("Detected NaN in target_values")

                if torch.isnan(pred_policies_logits).any():
                    logger.error("Detected NaN in pred_policies_logits")

                if torch.isnan(pred_values).any():
                    logger.error("Detected NaN in pred_values")

            p_loss = policy_loss(pred_policies_logits, target_policies.type(torch.float32))
            v_loss = value_loss(pred_values, target_values)
            loss = p_loss + v_loss

            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            num_batches += 1

        train_time = time.time() - train_start_time
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        logger.info(f"Training completed in {train_time:.2f}s")
        logger.info(f"Avg Loss: {avg_loss:.4f} "
                    f"(Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
        logger.info(f"Training used {num_batches} batches")

        total_iter_time = time.time() - iter_start_time
        logger.info(f"Iteration {iteration + 1} took {total_iter_time:.2f}s")

        # --- Save Checkpoint ---
        # Save a checkpoint after each iteration or at a specified interval
        if (iteration + 1) % 1 == 0 and args.should_checkpoint: # Save every iteration by default
            state = {
                'iteration': iteration + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(state, args.checkpoint_dir)

        if iteration % args.eval_interval == 0:
            logger.info("Beginning evaluation vs random player.")

            avg_score = 0
            avg_moves = 0
            avg_captures = 0
            for i in range(args.eval_iterations):
                eval_data = evaluate(net, RandomPlayer(game), args)
                avg_score += eval_data.score
                avg_moves += eval_data.moves
                avg_captures += eval_data.captures

            avg_score /= args.eval_iterations
            avg_moves /= args.eval_iterations
            avg_captures /= args.eval_iterations

            logger.info(f"Pente network score: {avg_score * 100}%")
            logger.info(f"Average moves: {avg_moves}")
            logger.info(f"Average captures: {avg_captures}")