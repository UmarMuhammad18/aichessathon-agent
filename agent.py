"""AI Chessathon agent: iterative deepening, alpha-beta, quiescence, PST, TT."""

from __future__ import annotations

import time
from typing import Optional

import chess

MATE, DRAW, INF = 30_000, 0, 40_000
PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
_MVV = [[0]*7 for _ in range(7)]
for v in range(1, 7):
    for a in range(1, 7):
        _MVV[v][a] = v*10 - a

PST_P_MG = [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0]
PST_P_EG = [0,0,0,0,0,0,0,0,80,80,80,80,80,80,80,80,50,50,50,50,50,50,50,50,30,30,30,30,30,30,30,30,20,20,20,20,20,20,20,20,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0]
PST_N = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
PST_B = [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20]
PST_R = [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0]
PST_Q = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20]
PST_K_MG = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]
PST_K_EG = [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,0,0,-10,-20,-30,-30,-10,20,30,30,20,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,20,30,30,20,-10,-30,-30,-30,0,0,0,0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50]

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
PHASE_W = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
MAX_PHASE = 24
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
TT_MAX = 1 << 18
_tt: dict = {}

def tt_get(key, depth, alpha, beta):
    e = _tt.get(key)
    if e is None: return None, None
    d, score, flag, mv = e
    if d < depth: return None, mv
    if flag == TT_EXACT: return score, mv
    if flag == TT_LOWER and score >= beta: return score, mv
    if flag == TT_UPPER and score <= alpha: return score, mv
    return None, mv

def tt_put(key, depth, score, flag, mv):
    if len(_tt) > TT_MAX: _tt.clear()
    _tt[key] = (depth, score, flag, mv)

def evaluate(board):
    if board.is_checkmate(): return -MATE
    if board.is_stalemate() or board.is_insufficient_material(): return DRAW
    mg, eg, phase = [0, 0], [0, 0], 0
    for sq, pc in board.piece_map().items():
        pt, col = pc.piece_type, (0 if pc.color == chess.WHITE else 1)
        val = PIECE_VALUE[pt]
        mw, ew, mb, eb = PST[pt]
        if col == 0:
            mg[0] += val + mw[sq]; eg[0] += val + ew[sq]
        else:
            mg[1] += val + mb[sq]; eg[1] += val + eb[sq]
        phase += PHASE_W[pt]
    phase = min(phase, MAX_PHASE)
    score = ((mg[0]-mg[1])*phase + (eg[0]-eg[1])*(MAX_PHASE-phase)) // MAX_PHASE
    return score if board.turn == chess.WHITE else -score

def order_moves(board, moves, tt_move):
    scored = []
    for m in moves:
        if m == tt_move: scored.append((10000, m)); continue
        if board.is_capture(m):
            vic = board.piece_type_at(m.to_square) or chess.PAWN
            att = board.piece_type_at(m.from_square) or chess.PAWN
            scored.append((1000 + _MVV[vic][att], m))
        elif m.promotion: scored.append((900 + m.promotion, m))
        else: scored.append((0, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]

_nodes, _deadline, _stop = 0, 0.0, False

def qsearch(board, alpha, beta):
    global _nodes, _stop
    _nodes += 1
    if time.perf_counter() >= _deadline:
        _stop = True; return alpha
    stand = evaluate(board)
    if stand >= beta: return beta
    if stand > alpha: alpha = stand
    moves = list(board.legal_moves) if board.is_check() else [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    for m in order_moves(board, moves, None):
        board.push(m); s = -qsearch(board, -beta, -alpha); board.pop()
        if _stop: return alpha
        if s >= beta: return beta
        if s > alpha: alpha = s
    return alpha

def negamax(board, depth, alpha, beta, ply):
    global _nodes, _stop
    _nodes += 1
    if time.perf_counter() >= _deadline:
        _stop = True; return alpha
    if board.is_repetition(2) or board.halfmove_clock >= 100: return DRAW
    key = board._transposition_key()
    hit, tt_mv = tt_get(key, depth, alpha, beta)
    if hit is not None: return hit
    if depth <= 0: return qsearch(board, alpha, beta)
    moves = list(board.legal_moves)
    if not moves: return -MATE + ply if board.is_check() else DRAW
    moves = order_moves(board, moves, tt_mv)
    best, best_mv, flag = -INF, None, TT_UPPER
    for m in moves:
        board.push(m); s = -negamax(board, depth-1, -beta, -alpha, ply+1); board.pop()
        if _stop: return alpha
        if s > best: best, best_mv = s, m
        if s > alpha:
            alpha, flag = s, TT_EXACT
            if alpha >= beta:
                flag = TT_LOWER; break
    tt_put(key, depth, best, flag, best_mv)
    return best

def search_root(board, time_ms):
    global _nodes, _deadline, _stop
    _nodes, _stop = 0, False
    soft = time_ms * 0.70 / 1000.0
    hard = max(0.04, (time_ms - 40) / 1000.0)
    t0 = time.perf_counter(); _deadline = t0 + hard
    moves = list(board.legal_moves)
    if len(moves) <= 1: return moves[0] if moves else chess.Move.null()
    best = moves[0]
    key = board._transposition_key()
    _, tt_mv = tt_get(key, 0, -INF, INF)
    if tt_mv is not None and tt_mv in moves: best = tt_mv
    for depth in range(1, 40):
        if time.perf_counter() - t0 >= soft and depth > 2: break
        _stop = False
        ordered = order_moves(board, moves, best)
        alpha, root_best, root_score = -INF, ordered[0], -INF
        for i, m in enumerate(ordered):
            board.push(m)
            if i == 0: s = -negamax(board, depth-1, -INF, -alpha, 1)
            else:
                s = -negamax(board, depth-1, -alpha-1, -alpha, 1)
                if s > alpha: s = -negamax(board, depth-1, -INF, -alpha, 1)
            board.pop()
            if _stop: break
            if s > root_score: root_score, root_best, alpha = s, m, s
        if not _stop:
            best = root_best
            if abs(root_score) > MATE - 500: break
        else: break
        if time.perf_counter() - t0 >= soft: break
    return best

def allocate_time(time_left_ms, move_n):
    usable = max(0, time_left_ms - 80)
    expected = max(12, 32 - move_n // 2)
    budget = usable // expected
    budget = min(budget, usable // 8)
    return max(80, min(budget, 10000))

_move_n = 0

def get_move(fen: str, time_left_ms: int) -> str:
    global _move_n
    board = chess.Board(fen)
    _move_n += 1
    if not board.legal_moves: return "0000"
    if time_left_ms < 100:
        for m in board.legal_moves:
            if board.is_capture(m) or m.promotion or board.gives_check(m): return m.uci()
        return next(iter(board.legal_moves)).uci()
    budget = allocate_time(time_left_ms, _move_n)
    return search_root(board, budget).uci()

_ = evaluate(chess.Board())
_ = order_moves(chess.Board(), list(chess.Board().legal_moves), None)
