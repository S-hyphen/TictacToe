import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import pygame

# Tic-Tac-Toe Environment
class TicTacToe:
    def __init__(self):
        self.board = np.full((3, 3), '_')
        self.turn = 'x'

    def reset(self):
        self.board = np.full((3, 3), '_')
        self.turn = 'x'
        return self.get_state()

    def get_state(self):
        mapping = {'x': 1, 'o': -1, '_': 0}
        state = np.vectorize(mapping.get)(self.board).flatten()
        return np.append(state, 1 if self.turn == 'x' else -1)

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == '_']

    def make_move(self, row, col):
        if self.board[row, col] == '_':
            self.board[row, col] = self.turn
            self.turn = 'o' if self.turn == 'x' else 'x'
            return True
        return False

    def check_winner(self):
        for line in np.vstack((self.board, self.board.T, [self.board.diagonal(), np.fliplr(self.board).diagonal()])):
            if np.all(line == 'x'):
                return 1
            elif np.all(line == 'o'):
                return -1
        return 0 if '_' not in self.board else None

# Neural Network Model
def create_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='linear')  # Output: row and column
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train Model with Self-Play
def train_model(model, episodes=5000):
    env = TicTacToe()
    for _ in range(episodes):
        env.reset()
        while True:
            state = env.get_state().reshape(1, -1)
            if random.random() < 0.2:  # Exploration
                move = random.choice(env.available_moves())
            else:  # Exploitation
                prediction = model.predict(state, verbose=0)[0]
                move = (round(prediction[0]), round(prediction[1]))
                if move not in env.available_moves():
                    move = random.choice(env.available_moves())
            env.make_move(*move)
            winner = env.check_winner()
            if winner is not None:
                break
    return model

# Pygame UI
def play_with_human(model):
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Tic-Tac-Toe AI")
    env = TicTacToe()
    running = True
    while running:
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                row, col = y // 100, x // 100
                if env.make_move(row, col):
                    state = env.get_state().reshape(1, -1)
                    prediction = model.predict(state, verbose=0)[0]
                    move = (round(prediction[0]), round(prediction[1]))
                    if move in env.available_moves():
                        env.make_move(*move)
        pygame.display.flip()
    pygame.quit()

# Train and Play
model = create_model()
model = train_model(model)
play_with_human(model)