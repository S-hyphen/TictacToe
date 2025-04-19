# TictacToe

In this project we impliment reinforcement learning to approximate the bestmoves in Tic-Tac-Toe.

This project uses a standard RL algorithm Q learning, and we've implemented it from scratch. It relies on basic self-play mechanism where the model is trained by repeatedly playing against itself and correctly identifying the best possible moves by looking up the Q-table which gets updated in every iteration.


**Install Required Dependencies**

In a virtual environment created using:
```bash
    python -m venv reinfocementLearning
```
Then change your source to the new virtual environment
```bash
    source reinforcementLearning/bin/activate
```

To run this project, ensure you have the necessary requirements setup.
```bash
    pip install -r requirements.txt
```
