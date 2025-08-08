import logging
import time
import torch
from torch.utils.data import DataLoader

from src.loss import policy_loss, value_loss
from src.mcts.mcts import play_game_with_mcts
from src.pente_dataloader import PenteDataset

logger = logging.getLogger(__name__)

def train(num_iterations: int, batch_games: int, eval_interval: int, net, optimizer):
    logger.info("Entered training loop")

    for iteration in range(num_iterations):
        iter_start_time = time.time()
        logger.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")

        sp_start_time = time.time()
        games = [play_game_with_mcts(net, num_simulations=400) for _ in range(batch_games)]
        sp_time = time.time() - sp_start_time
        logger.info(f"Generated {batch_games} games in {sp_time:.2f}s "
                    f"({batch_games / sp_time:.2f} games/sec)")

        examples = [(state, mcts_policy, outcome)
                    for game in games
                    for (state, mcts_policy, outcome) in game]
        logger.info(f"Collected {len(examples)} training positions")

        train_start_time = time.time()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        dataloader = DataLoader(PenteDataset(examples), batch_size=128, shuffle=True)
        for batch_idx, batch in enumerate(dataloader, 1):
            states, target_policies, target_values = batch
            pred_policies, pred_values = net(states)

            p_loss = policy_loss(pred_policies, target_policies)
            v_loss = value_loss(pred_values, target_values)
            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate stats
            total_loss += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0 or batch_idx == len(dataloader):
                logger.info(f"[Batch {batch_idx}/{len(dataloader)}] "
                            f"Loss: {loss.item():.4f} "
                            f"(Policy: {p_loss.item():.4f}, Value: {v_loss.item():.4f})")

        train_time = time.time() - train_start_time

        # Epoch summary
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        logger.info(f"Training completed in {train_time:.2f}s "
                    f"({num_batches / train_time:.2f} batches/sec)")
        logger.info(f"Avg Loss: {avg_loss:.4f} "
                    f"(Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")

        total_iter_time = time.time() - iter_start_time
        logger.info(f"Iteration {iteration + 1} took {total_iter_time:.2f}s")

        # --- Optional evaluation ---
        #if (iteration + 1) % eval_interval == 0:
        #    logger.info("Running evaluation...")
        #    evaluate_model(net)
