import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import pygame
import sys

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 600, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe AI Trainer")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
X_COLOR = (84, 84, 84)
O_COLOR = (242, 235, 211)

# Neural Network Model
def create_model():
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Game Logic
class TicTacToe:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.board = np.full((3, 3), '_')
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        
    def check_winner(self):
        lines = [
            *self.board,
            *self.board.T,
            np.diag(self.board),
            np.fliplr(self.board).diagonal()
        ]
        for line in lines:
            if len(set(line)) == 1 and line[0] != '_':
                return line[0]
        if '_' not in self.board:
            return 'draw'
        return None
    
    def make_move(self, row, col):
        if self.board[row][col] == '_' and not self.game_over:
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            self.winner = self.check_winner()
            if self.winner:
                self.game_over = True

# Encoding/Decoding functions
def encode_state(game):
    encoded = []
    for row in game.board:
        for cell in row:
            encoded.append(1 if cell == 'X' else -1 if cell == 'O' else 0)
    encoded.append(1 if game.current_player == 'X' else -1)
    return np.array(encoded)

def decode_move(move):
    return move // 3, move % 3

# Pygame Visualization
class GameVisualizer:
    def __init__(self, model):
        self.model = model
        self.game = TicTacToe()
        self.training = True
        self.font = pygame.font.Font(None, 36)
        
    def draw_board(self):
        screen.fill(BG_COLOR)
        # Draw grid lines
        pygame.draw.line(screen, LINE_COLOR, (200, 100), (200, 500), 15)
        pygame.draw.line(screen, LINE_COLOR, (400, 100), (400, 500), 15)
        pygame.draw.line(screen, LINE_COLOR, (0, 300), (600, 300), 15)
        pygame.draw.line(screen, LINE_COLOR, (0, 500), (600, 500), 15)
        
        # Draw moves
        for row in range(3):
            for col in range(3):
                cell = self.game.board[row][col]
                center = (col*200 + 100, row*200 + 200)
                if cell == 'X':
                    pygame.draw.line(screen, X_COLOR, (center[0]-50, center[1]-50), 
                                   (center[0]+50, center[1]+50), 15)
                    pygame.draw.line(screen, X_COLOR, (center[0]+50, center[1]-50), 
                                   (center[0]-50, center[1]+50), 15)
                elif cell == 'O':
                    pygame.draw.circle(screen, O_COLOR, center, 60, 15)
        
        # Draw status text
        if self.training:
            text = self.font.render("Training AI...", True, (255, 255, 255))
        else:
            status = f"Human's turn (X)" if self.game.current_player == 'X' else "AI's turn (O)"
            text = self.font.render(status, True, (255, 255, 255))
        screen.blit(text, (20, 20))
        
        pygame.display.flip()
    
    def train_ai(self, games=1000):
        training_states = []
        training_moves = []
        
        for game_num in range(games):
            self.game.reset()
            while not self.game.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                state = encode_state(self.game)
                valid_moves = [i for i in range(9) if self.game.board[i//3][i%3] == '_']
                
                if random.random() < 0.1:
                    move = random.choice(valid_moves)
                else:
                    pred = self.model.predict(state.reshape(1, -1), verbose=0)[0]
                    move = np.argmax([pred[i] if i in valid_moves else -np.inf for i in range(9)])
                
                training_states.append(state)
                training_moves.append(move)
                
                row, col = decode_move(move)
                self.game.make_move(row, col)
                self.draw_board()
                pygame.time.wait(50)  # Visualize training progress
                
            # Update training text
            text = self.font.render(f"Training Game {game_num+1}/{games}", True, (255, 255, 255))
            screen.blit(text, (20, 60))
            pygame.display.flip()
            
        return np.array(training_states), tf.keras.utils.to_categorical(training_moves, num_classes=9)
    
    def human_move(self, pos):
        x, y = pos
        col = x // 200
        row = (y - 100) // 200
        if 0 <= row < 3 and 0 <= col < 3:
            return row, col
        return None
    
    def run(self):
        # Initial training
        self.training = True
        X, y = self.train_ai(games=500)
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        self.training = False
        
        # Human vs AI game loop
        self.game.reset()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if not self.game.game_over and not self.training:
                    if self.game.current_player == 'X':  # Human turn
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            pos = pygame.mouse.get_pos()
                            move = self.human_move(pos)
                            if move and self.game.board[move[0]][move[1]] == '_':
                                self.game.make_move(*move)
                                self.draw_board()
                    else:  # AI turn
                        state = encode_state(self.game)
                        valid_moves = [i for i in range(9) if self.game.board[i//3][i%3] == '_']
                        pred = self.model.predict(state.reshape(1, -1), verbose=0)[0]
                        move = np.argmax([pred[i] if i in valid_moves else -np.inf for i in range(9)])
                        row, col = decode_move(move)
                        self.game.make_move(row, col)
                        self.draw_board()
                        pygame.time.wait(500)
            
            self.draw_board()
            clock.tick(30)

# Main execution
if __name__ == "__main__":
    model = create_model()
    visualizer = GameVisualizer(model)
    visualizer.run()