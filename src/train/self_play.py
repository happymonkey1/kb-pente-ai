import time

from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTS, MCTSArgs
from src.model.model_v1 import PenteNet
from src.game.game import Game
from src.pente_dataloader import PenteDataset

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import os

from src.train.arena import Arena
from src.train.nnet_player import NNetPlayer
from src.train.profession_game_loader import ProfessionGameLoader

logger = logging.getLogger(__name__)

@dataclass
class SelfPlayTrainerArgs:
    start_iteration: int
    professional_games_training_iterations: int
    self_play_training_iterations: int
    temp_threshold: float
    mcts_args: MCTSArgs
    professional_games_training_examples_filepath: str
    eval_iteration_interval: int = 1
    arena_num_games: int = 100
    batch_size: int = 8192
    batch_games: int = 1
    update_threshold: float = 0.6
    checkpoint_dir: str = "checkpoints"
    should_checkpoint: bool = True
    max_training_examples: int = 65536
    board_size: int = 19
    player_count: int = 2
    debug: bool = False
    should_use_arena: bool = True

@dataclass
class ModelTrainingStats:
    total_loss: float
    total_policy_loss: float
    total_value_loss: float
    num_batches: int

class SelfPlayTrainer:

    def __init__(
        self,
        game: 'Game',
        net: 'PenteNet',
        optimizer,
        device,
        args: SelfPlayTrainerArgs
    ):
        self.training_examples: list = []
        self.net = net
        self.game = game
        self.optimizer = optimizer
        self.device = device
        self.args = args

        self.policy_loss = torch.nn.CrossEntropyLoss()
        self.value_loss = torch.nn.MSELoss()

        self.previous_net = PenteNet.from_existing_model(self.net)

    def __play_game(self):
        examples = []
        board = self.game.init_board()
        current_player = Game.PLAYER_ONE
        step = 0
        mcts = MCTS(self.game, self.net, self.args.mcts_args)

        while True:
            step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            temp = int(step < self.args.temp_threshold)

            pi = mcts.get_action_prob(canonical_board, temp=temp)

            symmetries = self.game.get_symmetries(canonical_board, pi)
            for b, p in symmetries:
                examples.append([b, current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, next_player = self.game.apply_action(board, current_player, action)

            is_terminal, winner = self.game.check_game_end(board, current_player)
            if is_terminal:
                return [(x[0], x[2], winner) for x in examples]

            current_player = next_player

    def __build_self_play_training_examples(self):
        training_examples = []

        for game_index in range(self.args.batch_games):
            training_examples += self.__play_game()

        return training_examples

    def __load_training_examples(self):
        loader = ProfessionGameLoader(self.args.professional_games_training_examples_filepath, board_size=self.game.get_board_size())
        games = loader.load_games()
        return games

    def __get_training_examples_for_current_iteration(self, iteration: int):
        if self.args.professional_games_training_iterations > 0 and iteration < self.args.professional_games_training_iterations:
            if iteration == 0:
                logger.info("Loading training examples from professional games...")
                return self.__load_training_examples()
            else:
                return []
        else:
            logger.info("Generating training examples from self-play games...")
            return self.__build_self_play_training_examples()

    def __save_training_examples(self, training_examples):
        pass

    def train(self):
        logger.info("Entered training loop")
        total_iterations = self.args.professional_games_training_iterations + self.args.self_play_training_iterations
        for iteration in range(self.args.start_iteration, total_iterations):
            iter_start_time = time.time()
            logger.info(f"=== Iteration {iteration + 1}/{total_iterations} ===")

            sp_start_time = time.time()

            training_examples = self.__get_training_examples_for_current_iteration(iteration)
            sp_time = time.time() - sp_start_time
            logger.info(f"Generated {self.args.batch_games} training examples in {sp_time:.2f}s "
                        f"({self.args.batch_games / sp_time:.2f} examples/sec)")

            self.training_examples += training_examples

            if not self.training_examples:
                logger.warning("No training examples were generated in this iteration. Skipping training.")
                continue

            if 0 < self.args.max_training_examples < len(self.training_examples):
                logger.info(f"Removing {len(self.training_examples) - self.args.max_training_examples} training examples")
                self.training_examples = self.training_examples[-self.args.max_training_examples:]

            flat_examples = [item for item in training_examples]
            logger.info(f"Collected {len(flat_examples)} training positions ({len(flat_examples) / sp_time} positions/sec)")

            train_start_time = time.time()
            stats = self.__train_model(self.training_examples)
            train_time = time.time() - train_start_time

            avg_loss = stats.total_loss / stats.num_batches
            avg_policy_loss = stats.total_policy_loss / stats.num_batches
            avg_value_loss = stats.total_value_loss / stats.num_batches

            logger.info(f"Training completed in {train_time:.2f}s")
            logger.info(f"Avg Loss: {avg_loss:.4f} "
                        f"(Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
            logger.info(f"Training used {stats.num_batches} batches")

            total_iter_time = time.time() - iter_start_time
            logger.info(f"Iteration {iteration + 1} took {total_iter_time:.2f}s")

            if self.args.should_use_arena and self.args.arena_num_games > 0 and iteration >= self.args.professional_games_training_iterations:
                logger.info(f"Beginning Arena evaluation")

                # Load previous network from cached weights
                if self.args.should_checkpoint:
                    PenteNet.load_checkpoint(
                        checkpoint_dir=self.args.checkpoint_dir,
                        net=self.previous_net,
                        filename=self.previous_net.get_checkpoint_file_name(iteration - 1)
                    )

                logger.info("Initializing Arena")
                arena_start = time.time()
                previous_mcts = MCTS(self.game, self.previous_net, self.args.mcts_args)
                current_mcts = MCTS(self.game, self.net, self.args.mcts_args)
                arena = Arena(
                    player1=NNetPlayer(self.previous_net, previous_mcts),
                    player2=NNetPlayer(self.net, current_mcts),
                    game=self.game,
                    debug=self.args.debug,
                )
                arena_time = time.time() - arena_start
                logger.info(f"Arena initialized in {arena_time:.2f}s")

                arena_play_start = time.time()
                arena_stats = arena.play_games(self.args.arena_num_games)
                arena_play_time = time.time() - arena_play_start
                logger.info(f"Arena played {self.args.arena_num_games} games in {arena_play_time:.2f}s")
                previous_wins, current_wins, draws = arena_stats.p1_wins, arena_stats.p2_wins, arena_stats.draws

                logger.info(f"Arena stats:\n  previous model wins: {previous_wins}\n  current model wins: {current_wins}\n  draws: {draws}")
                win_rate = float(current_wins) / float(previous_wins + current_wins)
                logger.info(f"New model win rate: {win_rate:.2%}")
                avg_moves = arena_stats.avg_moves
                logger.info(f"Average moves: {avg_moves:.2f}")

                if previous_wins + current_wins == 0 or win_rate < self.args.update_threshold:
                    logger.info(f"Skipping model update due to low performance")
                    PenteNet.load_checkpoint(
                        checkpoint_dir=self.args.checkpoint_dir,
                        net=self.net,
                        filename=self.net.get_checkpoint_file_name(iteration)
                    )
                else:
                    logger.info(f"Accepting new model due to high performance")
                    training_state = {
                        'iteration': iteration + 1,
                        'state_dict': self.net.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    if self.args.should_checkpoint:
                        PenteNet.save_checkpoint(
                            state=training_state,
                            checkpoint_dir=self.args.checkpoint_dir,
                            filename=self.net.get_checkpoint_file_name(iteration + 1)
                        )
                        PenteNet.save_checkpoint(
                            state={
                                'state_dict': self.net.state_dict(),
                            },
                            checkpoint_dir=self.args.checkpoint_dir,
                            filename='best.pth.tar'
                        )
            elif self.args.should_checkpoint:
                logger.info("Arena is not enabled. Checkpointing model")
                training_state = {
                    'iteration': iteration + 1,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                PenteNet.save_checkpoint(
                    state=training_state,
                    checkpoint_dir=self.args.checkpoint_dir,
                    filename=self.net.get_checkpoint_file_name(iteration + 1)
                )

    def __train_model(self, training_examples: list[tuple[np.ndarray, np.ndarray, float]]) -> ModelTrainingStats:
        num_batches = 0
        dataloader = DataLoader(PenteDataset(training_examples), batch_size=self.args.batch_size, shuffle=True)
        total_loss, total_policy_loss, total_value_loss = 0, 0, 0

        logger.info(f"Training with {len(training_examples)} training positions")
        for batch_idx, batch in enumerate(dataloader, 1):
            states, target_policies, target_values = batch
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            pred_policies_logits, pred_values = self.net.forward(states)

            if self.args.debug:
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

            p_loss = self.policy_loss(pred_policies_logits, target_policies.type(dtype=torch.float32))
            v_loss = self.value_loss(pred_values, target_values)
            loss = p_loss + v_loss

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.net.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            num_batches += 1

        return ModelTrainingStats(total_loss, total_policy_loss, total_value_loss, num_batches)

