import numpy as np
import torch

from src.game.board_utils import legal_moves, apply_move, opponent, is_terminal, evaluate_terminal
from src.mcts.mcts_node import MCTSNode
from src.pente_dataloader import board_to_tensor
import torch.nn.functional as F

class MCTS:
    def __init__(self, net, num_simulations, c_puct=1.0):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, board, player):
        root = MCTSNode(prior = 0.0)

        policy_logits, _ = self.net(torch.from_numpy(board_to_tensor(board))).detach()
        policy_probs = F.softmax(policy_logits, dim=0).numpy()

        for move, prob in zip(legal_moves(board, player), policy_probs):
            root.children[move] = MCTSNode(prob)

        for _ in range(self.num_simulations):
            node, path = root, []
            cur_board, cur_player = board.copy(), player

            while node.children:
                move, node = max(
                    node.children.items(),
                    key=lambda item: self._puct(item[1], node)
                )
                path.append((cur_board, cur_player, move))
                cur_board = apply_move(cur_board, move, cur_player)
                cur_player = opponent(cur_player)

            terminated, winner = is_terminal(cur_board, cur_player)
            if terminated:
                value = evaluate_terminal(winner)
            else:
                policy_logits, value_tensor = self.net(
                    torch.from_numpy(board_to_tensor(cur_board))
                )
                value = value_tensor.item()
                # Add children for all legal moves
                for m, prob in zip(legal_moves(cur_board, cur_player),
                                   F.softmax(policy_logits, dim=0).detach().numpy()):
                    node.children[m] = MCTSNode(prior=prob)

            for b, p, m in reversed(path):
                node = b.children[m]
                node.value_sum += value * (1 if p == cur_player else -1)
                node.visit_count += 1
                value = -value

        visits = np.array([root.children[move].visit_count for move in legal_moves(board, player)], dtype=np.float32)
        policy = visits / visits.sum()

        return policy

    def _puct(self, child, parent):
        # Q + U
        q = child.value
        u = self.c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        return q + u

def play_game_with_mcts(net, num_simulations=400, max_moves=200, epsilon=0.1):
    board = np.zeros((19, 19), dtype=np.int32)
    player = 1
    examples = []
    captures = { 1: 0, 2: 0 }

    for _ in range(max_moves):
        policy = MCTS(net, num_simulations).run(board, player)

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

    outcome = evaluate_terminal(board)
    for i in range(len(examples)):
        player_of_move = 1 if i % 2 == 0 else 2
        examples[i] = (examples[i][0], examples[i][1], outcome * (1 if player_of_move == 1 else -1))

    return examples