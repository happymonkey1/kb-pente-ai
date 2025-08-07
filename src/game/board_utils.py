import numpy as np
from scipy.signal import convolve2d

def legal_moves(board, player):
    return list(zip(*np.where(board == 0)))

def apply_move(board, move, player, captures: dict[int, int]):
    r, c = move
    if board[r, c] != 0:
        raise ValueError(f"Cell ({r},{c}) is already occupied")

    new_board = board.copy()
    new_board[r, c] = player
    new_board, captured = _apply_captures(new_board, r, c, player)

    new_captures = captures.copy()
    new_captures[player] = captures.get(player, 0) + captured

    return new_board, new_captures

def opponent(player):
    return 1 if player == 2 else 2

def is_terminal(board: np.ndarray, captures: dict[int, int]):
    for p in (1, 2):
        if captures.get(p, 0) >= 10:
            return True, p

    for p in (1, 2):
        if _has_five_in_a_row_fast(board, p):
            return True, p

    if not np.any(board == 0):
        return True, 0

    return False, 0

def evaluate_terminal(winner):
    if winner == 0:
        return 0.0
    else:
        return 1.0 if winner == 1 else -1.0

def _apply_captures(board: np.ndarray, r: int, c: int, player: int):
    opp = opponent(player)
    H, W = board.shape
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    captured = 0

    for dx, dy in dirs:
        r1, c1 = r + dx, c + dy
        r2, c2 = r + 2 * dx, c + 2 * dy
        r3, c3 = r + 3 * dx, c + 3 * dy

        if (
                _in_bounds(r1, c1, H, W)
                and _in_bounds(r2, c2, H, W)
                and _in_bounds(r3, c3, H, W)
                and board[r1, c1] == opp
                and board[r2, c2] == opp
                and board[r3, c3] == player
        ):
            board[r1, c1] = 0
            board[r2, c2] = 0
            captured += 1


        r1, c1 = r - dx, c - dy
        r2, c2 = r - 2 * dx, c - 2 * dy
        r3, c3 = r - 3 * dx, c - 3 * dy

        if (
                _in_bounds(r1, c1, H, W)
                and _in_bounds(r2, c2, H, W)
                and _in_bounds(r3, c3, H, W)
                and board[r1, c1] == opp
                and board[r2, c2] == opp
                and board[r3, c3] == player
        ):
            board[r1, c1] = 0
            board[r2, c2] = 0
            captured += 1

    return board, captured

def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w

def _has_five_in_a_row(board: np.ndarray, player: int) -> bool:
    H, W = board.shape
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in dirs:
        for r in range(H):
            for c in range(W):
                if board[r, c] != player:
                    continue

                for k in range(1, 5):
                    nr, nc = r + k * dr, c + k * dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        break
                    if board[nr, nc] != player:
                        break
                else:
                    return True
    return False

def _has_five_in_a_row_fast(board: np.ndarray, player: int) -> bool:
    mask = (board == player).astype(np.int8)
    kernel = np.ones((5, 1), dtype=np.int8)

    if np.any(convolve2d(mask, kernel.T, mode='valid') == 5):
        return True

    if np.any(convolve2d(mask, kernel, mode='valid') == 5):
        return True

    diag_kernel = np.eye(5, dtype=np.int8)
    if np.any(convolve2d(mask, diag_kernel, mode='valid') == 5):
        return True

    diag_kernel_lr = np.fliplr(diag_kernel)
    if np.any(convolve2d(mask, diag_kernel_lr, mode='valid') == 5):
        return True

    return False