"""AI Chessathon agent v3 entrypoint."""
from __future__ import annotations

import chess

import engine
from engine import allocate_time, evaluate, order_moves, search_root


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    engine._move_n += 1

    if not board.legal_moves:
        return "0000"

    if time_left_ms < 150:
        for m in board.legal_moves:
            if board.is_capture(m) or m.promotion or board.gives_check(m):
                return m.uci()
        return next(iter(board.legal_moves)).uci()

    budget = allocate_time(time_left_ms, engine._move_n)
    return search_root(board, budget).uci()


# Warm-up in the 90s init budget
_ = evaluate(chess.Board())
_ = order_moves(chess.Board(), list(chess.Board().legal_moves), None, 0)
