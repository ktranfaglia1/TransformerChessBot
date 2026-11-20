# Transformer Chess Bot

**A chess AI that pairs a transformer-style position evaluator with Monte Carlo Tree Search to play more human-like chess, with a desktop GUI that shows its evaluation live.**

[Final report (PDF)](TranfagliaObrien_2025_FinalReport.pdf)

Engines like Stockfish play deterministic, superhuman chess. This research project, the final project for COSC 490 (Large Language Models) at Salisbury University in spring 2025, asked whether techniques from language models could produce play that feels more human. It was inspired by AlphaZero:

- A PyTorch network trained on human games predicts how a position will end: White wins, draw, or Black wins.
- Monte Carlo Tree Search (MCTS) uses those predictions to choose moves.
- A PySide6 GUI lets you play against the bot, watch two bots play each other, and see the model's evaluation update after every move.

## Results

- **Validation accuracy of about 91%** when predicting game outcomes from positions in the grandmaster dataset (training and validation curves are in [`results/graphs/`](results/graphs)).
- **Playing strength of about 400 Elo,** well short of our 2,200 target. The model learned basic patterns and its play varied in a human-like way, but it often missed tactics such as undefended pieces and immediate threats.
- **The main lesson: data quality.**
  - Grandmaster games are high quality but rarely show mistakes being punished.
  - Lower-rated games contain those lessons, but also a lot of noise.
  - A hybrid dataset helped. Memory limits on our hardware capped how much data and how large a model we could train.

The [report](TranfagliaObrien_2025_FinalReport.pdf) covers the method, the results, and future directions: self-play reinforcement learning, policy/value heads for MCTS, and training distributed across sharded data.

## How it works

### Position encoding
Each board is encoded as an **837-dimensional** one-hot tensor covering piece placement, castling rights and other game state, following published chess-encoding research.

### Model (`bigger_model.py`)
A transformer-inspired network:

1. **Input block:** layer normalization, a two-layer MLP that expands then projects, and a residual skip projection.
2. **Hidden blocks:** a stack of **multi-head self-attention** layers, each followed by a feed-forward network, with residual connections and pre-layer normalization. (Putting normalization first rather than last made a large difference in our experiments.)
3. **Output block:** projects to three probabilities: White win, draw, Black win.

The default configuration is 512 wide, 16 blocks deep, with 4 attention heads.

Training details:
- Xavier initialization, GELU activations and the Adam optimizer.
- A learning-rate schedule (ReduceLROnPlateau), gradient clipping and early stopping.
- **Position averaging:** when the same position appears in many games, its outcomes are averaged. This resolves contradictory labels.

The original baseline MLP is kept in `train/model.py` for comparison.

### Search (`main.py`)
MCTS selects children with UCT (exploration weight √2) and scores positions with the model's predicted outcome. It runs a fixed number of iterations per move, which keeps response time predictable.

### Data
Training games came from Lichess and grandmaster game collections. `dataTester.py` handles cleaning:

- **Parsing:** reads PGN files into pandas.
- **Filtering:** removes illegal moves and short grandmaster draws (under 30 moves), and can restrict the opening repertoire.
- **Balancing:** evens out wins, losses and draws.
- **Variants:** produces datasets with different Elo floors.

`dataset.py` converts games into tensor datasets.

## The GUI

Built with PySide6 and `python-chess`:

- **Players:** set White and Black to Human or AI independently, so you can play human vs. human, human vs. bot, or bot vs. bot.
- **Live evaluation:** a panel shows White win, draw and Black win probabilities, updated after every move.
- **Board:** click-to-move with legal-move highlighting, a scrolling move history, game-status messages (check, checkmate, draws such as threefold repetition), undo, board flip and new game.
- **Responsiveness:** AI moves are computed on a background thread, so the interface never freezes.

## Running it

Install PyTorch for your platform (see [pytorch.org](https://pytorch.org/get-started/locally/); use the CUDA build if you have an NVIDIA GPU), then:

```bash
pip install python-chess PySide6 pandas numpy scikit-learn tqdm matplotlib
python gpu_check.py      # optional: confirm PyTorch can see your GPU
python main.py           # launch the GUI
```

- **Model weights:** trained weights (`*.pth`) are too large for the repository and are git-ignored. Without `models/enhanced_chess_model.pth`, the GUI still runs, using an untrained model or a simple fallback evaluation.
- **Training your own:** run `python bigger_model.py`. [`train/bigger_model_guide.md`](train/bigger_model_guide.md) walks through the configuration.
- **Large files:** the CSV datasets and zipped tensor sets are stored with **Git LFS**. Run `git lfs install` before cloning to download them.

## Repository structure

```
├── main.py                  # GUI, MCTS, and model integration (entry point)
├── bigger_model.py          # Enhanced attention-based model: architecture, training, inference
├── dataset.py               # Converts games to tensors
├── dataTester.py            # PGN parsing, cleaning, filtering and balancing
├── chessGame.py             # Standalone two-player GUI (no AI)
├── gpu_check.py             # CUDA availability check
├── convert_small.py         # Builds the small dataset sample
├── train/                   # Training copies, baseline MLP (model.py), and the model guide
├── data/                    # Lichess and GM game datasets (Git LFS)
├── models/                  # Saved baseline weights (zipped, Git LFS)
├── results/                 # Training metrics (CSV) and loss/accuracy plots
└── TranfagliaObrien_2025_FinalReport.pdf
```

## Authors

| Name | Contributions |
|---|---|
| [Kyle Tranfaglia](https://github.com/ktranfaglia1) | Built the complete GUI and evaluation display; built the larger attention-based model (self-attention blocks, residual connections, feed-forward networks, layer normalization), the MCTS integration and position averaging; assembled the Lichess, GM, balanced and Elo-filtered datasets; tested model weights in the app |
| [Dustin O'Brien](https://github.com/Omniladder) | Designed and trained the neural network variants (initial MLP and the attention architectures); recommended MCTS; implemented Xavier initialization, GELU and Adam; wrote the PGN-to-tensor pipeline |
