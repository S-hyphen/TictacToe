import numpy as np
import tensorflow as tf
import random
import pygame
import sys
from collections import deque

# Configuration
SCREEN_SIZE = 600
GRID_SIZE = 3
CELL_SIZE = SCREEN_SIZE // GRID_SIZE

# Hyperparameters
TRAINING_GAMES = 5000
EXPLORATION_START = 1.0
EXPLORATION_END = 0.01
EXPLORATION_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 100000

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
X_COLOR = (84, 84, 84)
O_COLOR = (242, 235, 211)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Tic Tac Toe AI")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.full((3, 3), '_')
        self.current_player = 'X'
        self.winner = None
        self.game_over = False
    
    def check_winner(self):
        lines = [
            *self.board,
            *self.board.T,
            np.diag(self.board),
            np.diag(np.fliplr(self.board))
        ]
        for line in lines:
            if len(set(line)) == 1 and line[0] != '_':
                return line[0]
        return 'draw' if '_' not in self.board else None
    
    def make_move(self, row, col):
        if self.board[row][col] == '_' and not self.game_over:
            self.board[row][col] = self.current_player
            self.winner = self.check_winner()
            self.game_over = self.winner is not None
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

class DQNAgent:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.exploration_rate = EXPLORATION_START
        self.update_target_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(9, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_moves):
        if np.random.rand() < self.exploration_rate:
            return random.choice(valid_moves)
        state = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state, verbose=0)[0]
        return np.argmax([act_values[i] if i in valid_moves else -np.inf for i in range(9)])
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([x[0] for x in minibatch])
        targets = self.model.predict(states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                target = reward + 0.95 * np.max(self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            targets[i][action] = target
        
        self.model.fit(states, targets, batch_size=BATCH_SIZE, verbose=0)
        self.exploration_rate = max(EXPLORATION_END, self.exploration_rate * EXPLORATION_DECAY)

def encode_state(game):
    encoded = []
    for row in game.board:
        for cell in row:
            encoded.append(1 if cell == 'X' else -1 if cell == 'O' else 0)
    encoded.append(1 if game.current_player == 'X' else -1)
    return np.array(encoded)

def decode_move(move):
    return move // 3, move % 3

def draw_board(game):
    screen.fill(BG_COLOR)
    
    # Draw grid lines
    for i in range(1, GRID_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (i*CELL_SIZE, 0), (i*CELL_SIZE, SCREEN_SIZE), 15)
        pygame.draw.line(screen, LINE_COLOR, (0, i*CELL_SIZE), (SCREEN_SIZE, i*CELL_SIZE), 15)
    
    # Draw X's and O's
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            center = (col*CELL_SIZE + CELL_SIZE//2, row*CELL_SIZE + CELL_SIZE//2)
            if game.board[row][col] == 'X':
                pygame.draw.line(screen, X_COLOR, (center[0]-50, center[1]-50), 
                               (center[0]+50, center[1]+50), 15)
                pygame.draw.line(screen, X_COLOR, (center[0]+50, center[1]-50), 
                               (center[0]-50, center[1]+50), 15)
            elif game.board[row][col] == 'O':
                pygame.draw.circle(screen, O_COLOR, center, 60, 15)
    
    # Display status
    if not game.game_over:
        text = font.render(f"{game.current_player}'s Turn", True, (255, 255, 255))
        screen.blit(text, (20, 20))
    pygame.display.flip()

def train_ai(agent, episodes):
    game = TicTacToe()
    
    for episode in range(episodes):
        game.reset()
        state = encode_state(game)
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            valid_moves = [i for i in range(9) if game.board[i//3][i%3] == '_']
            action = agent.act(state, valid_moves)
            row, col = decode_move(action)
            
            prev_state = state
            game.make_move(row, col)
            next_state = encode_state(game)
            done = game.game_over
            
            # Calculate reward
            if game.winner == 'X':
                reward = 1.0
            elif game.winner == 'O':
                reward = -1.0
            elif game.winner == 'draw':
                reward = 0.5
            else:
                reward = 0.1  # Small reward for valid moves
            
            agent.remember(prev_state, action, reward, next_state, done)
            state = next_state
            
            agent.replay()
            
            if episode % 100 == 0:
                draw_board(game)
                pygame.time.wait(10)
        
        if episode % 100 == 0:
            agent.update_target_model()
            print(f"Episode: {episode+1}, Exploration: {agent.exploration_rate:.2f}")

def human_vs_ai(agent):
    game = TicTacToe()
    agent.exploration_rate = 0.0  # Disable exploration
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if game.game_over:
                game.reset()
                continue
            
            if game.current_player == 'X':  # Human
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    if game.make_move(row, col):
                        draw_board(game)
            
            else:  # AI
                state = encode_state(game)
                valid_moves = [i for i in range(9) if game.board[i//3][i%3] == '_']
                action = agent.act(state, valid_moves)
                row, col = decode_move(action)
                game.make_move(row, col)
                draw_board(game)
                pygame.time.wait(500)
        
        draw_board(game)
        clock.tick(30)

if __name__ == "__main__":
    # Training phase
    agent = DQNAgent()
    print("Training AI...")
    train_ai(agent, TRAINING_GAMES)
    
    # Save trained model
    agent.model.save('tictactoe_ai.h5')
    
    # Human vs AI
    print("Starting game vs Human")
    human_vs_ai(agent)