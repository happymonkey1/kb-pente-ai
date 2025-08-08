from torch.utils.data import DataLoader

from src.loss import policy_loss, value_loss
from src.mcts.mcts import play_game_with_mcts
from src.pente_dataloader import PenteDataset
import logging

logger = logging.getLogger(__name__)

def train(num_iterations: int, batch_games: int, eval_interval: int, net, optimizer):
    logger.info("Entered training loop")
    for iteration in range(num_iterations):
        if iteration % 100 == 0:
            logger.info(f"Iteration {iteration}")

        games = []
        for _ in range(batch_games):
            game = play_game_with_mcts(net, num_simulations=400)
            games.append(game)

        examples = []
        for game in games:
            for state, mcts_policy, outcome in game:
                examples.append((state, mcts_policy, outcome))

        for batch in DataLoader(PenteDataset(examples), batch_size=128, shuffle=True):
            states, target_policies, target_values = batch
            pred_policies, pred_values = net(states)
            loss = policy_loss(pred_policies, target_policies) + value_loss(pred_values, target_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #if iteration % eval_interval == 0:
        #    evaluate(net)