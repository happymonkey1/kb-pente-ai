import logging

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from src.game.board_utils import (
    legal_moves, apply_move, opponent, is_terminal, evaluate_terminal, board_to_tensor
)
from src.mcts.mcts_node import MCTSNode

logger = logging.getLogger(__name__)

class MCTS:
    def __init__(self, net: nn.Module, num_simulations: int, c_puct=1.0):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, board: np.ndarray, player: int, captures: dict[int, int]):
        logger.debug("Entered MCTS.run()")

        root = MCTSNode(
            prior=1.0,
            board=board,
            player=player,
            captures=captures.copy(),
            parent=None,
        )
        policy_logits, _ = self.net(board_to_tensor(board))
        policy_probs = F.softmax(policy_logits, dim=0).detach().cpu().numpy()
        cur_captures = captures.copy()

        for move, prob in zip(legal_moves(board, player), policy_probs):
            root.children[move] = MCTSNode(
                prior=prob,
                board=board,
                player=player,
                captures=captures.copy(),
                parent=root,
            )

        for i in range(self.num_simulations):
            logger.debug(f"MCTS simulation step: {i + 1}/{self.num_simulations}")
            node: 'MCTSNode' = root
            path: list[tuple['MCTSNode', tuple[int, int]]] = []
            cur_board, cur_player = board.copy(), player

            logger.debug(f"len(node.children) = {len(node.children)}")
            while node.children:
                move, new_node = max(
                    node.children.items(),
                    key=lambda item: self._puct(item[1], node),
                )

                path.append((node, move))
                cur_board, cur_captures = apply_move(
                    new_node.board,
                    move,
                    new_node.player,
                    new_node.captures,
                )
                cur_player = opponent(cur_player)

                node = new_node

            terminated, winner = is_terminal(cur_board, node.captures)
            if terminated:
                value = evaluate_terminal(winner)
            else:
                policy_logits, value_tensor = self.net(board_to_tensor(cur_board))
                value = value_tensor.item()

                for m, prob in zip(
                        legal_moves(cur_board, cur_player),
                        F.softmax(policy_logits, dim=0).detach().cpu().numpy(),
                ):
                    node.children[m] = MCTSNode(
                        prior=prob,
                        board=cur_board,
                        player=cur_player,
                        captures=cur_captures.copy(),
                        parent=node,
                    )

            logger.debug(f"Generated len(path) = {len(path)}")
            for b, move in reversed(path):
                logger.debug(f"Updating node {b} at move {move}")
                node = b.children[move]
                node.value_sum += value
                node.visit_count += 1
                value = -value

        visits = np.array(
            [root.children[move].visit_count for move in legal_moves(board, player)],
            dtype=np.float32,
        )
        policy = visits / visits.sum()

        return policy

    def _puct(self, child: MCTSNode, parent: MCTSNode):
        q = child.value
        u = (
                self.c_puct
                * child.prior
                * np.sqrt(parent.visit_count)
                / (1 + child.visit_count)
        )

        return q + u


def play_game_with_mcts(net, num_simulations=400, max_moves=200, epsilon=0.1):
    board = np.zeros((19, 19), dtype=np.int32)
    player = 1
    examples = []
    captures = { 1: 0, 2: 0 }
    winner = 0

    for _ in range(max_moves):
        policy = MCTS(net, num_simulations).run(board, player, captures)

        if np.random.rand() < epsilon:
            move = tuple(np.random.choice((a for a in legal_moves(board, player)), 1)[0])
        else:
            legal = legal_moves(board, player)
            move = legal[np.argmax([policy[legal.index(m)] for m in legal])]

        state_tensor = board_to_tensor(board)   # (2, H, W)
        examples.append((state_tensor, policy, None))  # value to be filled later

        board, captured = apply_move(board, move, player, captures)
        player = opponent(player)

        terminated, winner = is_terminal(board, captured)
        if terminated:
            break

    outcome = evaluate_terminal(winner)
    for i in range(len(examples)):
        player_of_move = 1 if i % 2 == 0 else 2
        examples[i] = (examples[i][0], examples[i][1], outcome * (1 if player_of_move == 1 else -1))

    return examples