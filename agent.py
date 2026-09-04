"""AI Chessathon agent v2 — stronger classical search.

Upgrades vs v1:
  - Contempt when ahead (prefer play over soft draws)
  - Root anti-repetition when eval clearly positive
  - Aspiration windows on iterative deepening
  - Check extension (+1 ply when in check)
  - Killer moves + history heuristic for quiet ordering
  - Stronger eval: bishop pair, tempo, simple pawn structure
  - Delta pruning in quiescence
  - More aggressive, still safe, time allocation
  - TT survives across moves within a game
"""

from __future__ import annotations

import time
from typing import Optional

import chess

MATE = 30_000
DRAW = 0
INF = 40_000
CONTEMPT = 25

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

_MVV = [[0] * 7 for _ in range(7)]
for v in range(1, 7):
    for a in range(1, 7):
        _MVV[v][a] = v * 10 - a

PST_P_MG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
PST_P_EG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
    0, 0, 0, 0, 0, 0, 0, 0,
]
PST_N = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
PST_B = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
PST_R = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]
PST_Q = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
PST_K_MG = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
PST_K_EG = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]


def _mir(t):
    return [t[s ^ 56] for s in range(64)]


PST = {
    chess.PAWN: (PST_P_MG, PST_P_EG, _mir(PST_P_MG), _mir(PST_P_EG)),
    chess.KNIGHT: (PST_N, PST_N, _mir(PST_N), _mir(PST_N)),
    chess.BISHOP: (PST_B, PST_B, _mir(PST_B), _mir(PST_B)),
    chess.ROOK: (PST_R, PST_R, _mir(PST_R), _mir(PST_R)),
    chess.QUEEN: (PST_Q, PST_Q, _mir(PST_Q), _mir(PST_Q)),
    chess.KING: (PST_K_MG, PST_K_EG, _mir(PST_K_MG), _mir(PST_K_EG)),
}

PHASE_W = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
MAX_PHASE = 24

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
TT_MAX = 1 << 18
_tt = {}
_killers = [[None, None] for _ in range(128)]
_history = [[0] * 64 for _ in range(64)]


def tt_get(key, depth, alpha, beta):
    e = _tt.get(key)
    if e is None:
        return None, None
    d, score, flag, mv = e
    if d < depth:
        return None, mv
    if flag == TT_EXACT:
        return score, mv
    if flag == TT_LOWER and score >= beta:
        return score, mv
    if flag == TT_UPPER and score <= alpha:
        return score, mv
    return None, mv


def tt_put(key, depth, score, flag, mv):
    if len(_tt) > TT_MAX:
        _tt.clear()
    _tt[key] = (depth, score, flag, mv)


def _record_killer(ply, move):
    if ply >= len(_killers):
        return
    if _killers[ply][0] != move:
        _killers[ply][1] = _killers[ply][0]
        _killers[ply][0] = move


def _record_history(move, depth):
    _history[move.from_square][move.to_square] += depth * depth


def evaluate(board):
    if board.is_checkmate():
        return -MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return DRAW

    mg = [0, 0]
    eg = [0, 0]
    phase = 0
    bishops = [0, 0]
    pawns_w = [0] * 8
    pawns_b = [0] * 8

    for sq, pc in board.piece_map().items():
        pt = pc.piece_type
        col = 0 if pc.color == chess.WHITE else 1
        val = PIECE_VALUE[pt]
        mw, ew, mb, eb = PST[pt]
        if col == 0:
            mg[0] += val + mw[sq]
            eg[0] += val + ew[sq]
            if pt == chess.BISHOP:
                bishops[0] += 1
            if pt == chess.PAWN:
                pawns_w[chess.square_file(sq)] += 1
        else:
            mg[1] += val + mb[sq]
            eg[1] += val + eb[sq]
            if pt == chess.BISHOP:
                bishops[1] += 1
            if pt == chess.PAWN:
                pawns_b[chess.square_file(sq)] += 1
        phase += PHASE_W[pt]

    if bishops[0] >= 2:
        mg[0] += 30
        eg[0] += 40
    if bishops[1] >= 2:
        mg[1] += 30
        eg[1] += 40

    for f in range(8):
        if pawns_w[f] > 1:
            mg[0] -= 12 * (pawns_w[f] - 1)
            eg[0] -= 18 * (pawns_w[f] - 1)
        if pawns_b[f] > 1:
            mg[1] -= 12 * (pawns_b[f] - 1)
            eg[1] -= 18 * (pawns_b[f] - 1)

    phase = min(phase, MAX_PHASE)
    score = ((mg[0] - mg[1]) * phase + (eg[0] - eg[1]) * (MAX_PHASE - phase)) // MAX_PHASE
    score += 10 if board.turn == chess.WHITE else -10
    stm = score if board.turn == chess.WHITE else -score

    if stm > 80:
        stm += CONTEMPT
    elif stm < -80:
        stm -= CONTEMPT
    return stm


def order_moves(board, moves, tt_move, ply=0):
    scored = []
    k0 = _killers[ply][0] if ply < len(_killers) else None
    k1 = _killers[ply][1] if ply < len(_killers) else None
    for m in moves:
        if m == tt_move:
            scored.append((1_000_000, m))
            continue
        if board.is_capture(m):
            vic = board.piece_type_at(m.to_square) or chess.PAWN
            att = board.piece_type_at(m.from_square) or chess.PAWN
            scored.append((100_000 + _MVV[vic][att] * 100, m))
        elif m.promotion:
            scored.append((90_000 + m.promotion, m))
        elif m == k0:
            scored.append((80_000, m))
        elif m == k1:
            scored.append((70_000, m))
        else:
            scored.append((_history[m.from_square][m.to_square], m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


_nodes = 0
_deadline = 0.0
_stop = False


def qsearch(board, alpha, beta):
    global _nodes, _stop
    _nodes += 1
    if time.perf_counter() >= _deadline:
        _stop = True
        return alpha

    stand = evaluate(board)
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand

    if board.is_check():
        moves = list(board.legal_moves)
    else:
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        filtered = []
        for m in moves:
            vic = board.piece_type_at(m.to_square)
            gain = PIECE_VALUE.get(vic, 100) if vic else 100
            if m.promotion:
                gain += PIECE_VALUE.get(m.promotion, 800) - 100
            if stand + gain + 200 >= alpha:
                filtered.append(m)
        moves = filtered

    for m in order_moves(board, moves, None, 0):
        board.push(m)
        s = -qsearch(board, -beta, -alpha)
        board.pop()
        if _stop:
            return alpha
        if s >= beta:
            return beta
        if s > alpha:
            alpha = s
    return alpha


def negamax(board, depth, alpha, beta, ply):
    global _nodes, _stop
    _nodes += 1
    if time.perf_counter() >= _deadline:
        _stop = True
        return alpha

    if board.is_repetition(2) or board.halfmove_clock >= 100:
        return DRAW

    key = board._transposition_key()
    hit, tt_mv = tt_get(key, depth, alpha, beta)
    if hit is not None:
        return hit

    in_check = board.is_check()
    if in_check and depth > 0:
        depth += 1

    if depth <= 0:
        return qsearch(board, alpha, beta)

    moves = list(board.legal_moves)
    if not moves:
        return -MATE + ply if in_check else DRAW

    moves = order_moves(board, moves, tt_mv, ply)
    best = -INF
    best_mv = None
    flag = TT_UPPER

    for i, m in enumerate(moves):
        is_cap = board.is_capture(m) or m.promotion
        board.push(m)
        gives_chk = board.is_check()
        reduced = False
        if depth >= 3 and i >= 4 and not in_check and not is_cap and not gives_chk:
            s = -negamax(board, depth - 2, -alpha - 1, -alpha, ply + 1)
            reduced = True
        else:
            s = -INF

        if not reduced or s > alpha:
            if i == 0:
                s = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                s = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < s < beta:
                    s = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if _stop:
            return alpha

        if s > best:
            best = s
            best_mv = m
        if s > alpha:
            alpha = s
            flag = TT_EXACT
            if alpha >= beta:
                flag = TT_LOWER
                if not is_cap:
                    _record_killer(ply, m)
                    _record_history(m, depth)
                break

    tt_put(key, depth, best, flag, best_mv)
    return best


def search_root(board, time_ms):
    global _nodes, _deadline, _stop
    _nodes = 0
    _stop = False
    soft = time_ms * 0.82 / 1000.0
    hard = max(0.05, (time_ms - 35) / 1000.0)
    t0 = time.perf_counter()
    _deadline = t0 + hard

    moves = list(board.legal_moves)
    if not moves:
        return chess.Move.null()
    if len(moves) == 1:
        return moves[0]

    static = evaluate(board)
    candidates = moves
    if static > 120:
        non_rep = []
        for m in moves:
            board.push(m)
            rep = board.is_repetition(2)
            board.pop()
            if not rep:
                non_rep.append(m)
        if non_rep:
            candidates = non_rep

    best = candidates[0]
    key = board._transposition_key()
    _, tt_mv = tt_get(key, 0, -INF, INF)
    if tt_mv is not None and tt_mv in candidates:
        best = tt_mv

    alpha_window = 40
    for depth in range(1, 48):
        if time.perf_counter() - t0 >= soft and depth > 3:
            break
        _stop = False
        ordered = order_moves(board, candidates, best, 0)

        if depth >= 3 and abs(static) < MATE - 1000:
            a = static - alpha_window
            b = static + alpha_window
        else:
            a, b = -INF, INF

        root_best = ordered[0]
        root_score = -INF
        failed = False

        for i, m in enumerate(ordered):
            board.push(m)
            if i == 0:
                s = -negamax(board, depth - 1, -b, -a, 1)
            else:
                s = -negamax(board, depth - 1, -a - 1, -a, 1)
                if s > a:
                    s = -negamax(board, depth - 1, -b, -a, 1)
            board.pop()
            if _stop:
                break
            if s > root_score:
                root_score = s
                root_best = m
                a = max(a, s)
            if depth >= 3 and (s <= static - alpha_window or s >= static + alpha_window):
                failed = True

        if not _stop:
            best = root_best
            static = root_score
            if abs(root_score) > MATE - 500:
                break
            alpha_window = min(400, alpha_window * 2) if failed else 40
        else:
            break

        if time.perf_counter() - t0 >= soft:
            break

    return best


def allocate_time(time_left_ms, move_n):
    usable = max(0, time_left_ms - 100)
    expected = max(8, 28 - move_n // 2)
    budget = usable // expected
    budget = min(budget, max(120, usable // 6))
    return max(120, min(budget, 12_000))


_move_n = 0


def get_move(fen: str, time_left_ms: int) -> str:
    global _move_n
    board = chess.Board(fen)
    _move_n += 1

    if not board.legal_moves:
        return "0000"

    if time_left_ms < 120:
        for m in board.legal_moves:
            if board.is_capture(m) or m.promotion or board.gives_check(m):
                return m.uci()
        return next(iter(board.legal_moves)).uci()

    budget = allocate_time(time_left_ms, _move_n)
    return search_root(board, budget).uci()


_ = evaluate(chess.Board())
_ = order_moves(chess.Board(), list(chess.Board().legal_moves), None, 0)
