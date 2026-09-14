# aichessathon-agent

Classical search agent for [AI Chessathon](https://aichessathon.com) (Optiver / Encode Club).


## Quick start

```bash
make setup
make play      # one game vs greedy, real clock
make arena     # 20 fast games
make zip       # submission.zip with agent.py at root
```

## Agent

`get_move(fen, time_left_ms) -> uci` with iterative deepening, alpha-beta, quiescence, MG/EG piece-square tables, MVV-LVA + TT ordering, and clock-aware time management.

Local arena (approx): greedy 100%, minimax ~87.5%.
