import numpy as np
import tensorflow as tf
import random
import pygame
import sys

# Configure TensorFlow to use GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Initialize Pygame modules explicitly
pygame.init()
pygame.font.init()
pygame.display.init()

WIDTH, HEIGHT = 600, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe AI Trainer")
clock = pygame.time.Clock()

# Colors and Constants
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
X_COLOR = (84, 84, 84)
O_COLOR = (242, 235, 211)
TRAINING_GAMES = 500  # Reduced for initial testing

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

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

def encode_state(game):
    encoded = []
    for row in game.board:
        for cell in row:
            encoded.append(1 if cell == 'X' else -1 if cell == 'O' else 0)
    encoded.append(1 if game.current_player == 'X' else -1)
    return np.array(encoded)

def decode_move(move):
    return move // 3, move % 3

class GameVisualizer:
    def __init__(self, model):
        self.model = model
        self.game = TicTacToe()
        self.training = True
        try:
            self.font = pygame.font.Font(None, 36)
        except:
            self.font = pygame.font.SysFont('arial', 36)
        
    def draw_board(self):
        try:
            screen.fill(BG_COLOR)
            # Draw grid lines
            pygame.draw.line(screen, LINE_COLOR, (200, 100), (200, 700), 15)
            pygame.draw.line(screen, LINE_COLOR, (400, 100), (400, 700), 15)
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
            
            # Status text
            if self.training:
                text = self.font.render("Training AI...", True, (255, 255, 255))
            else:
                status = "Human's turn (X)" if self.game.current_player == 'X' else "AI's turn (O)"
                text = self.font.render(status, True, (255, 255, 255))
            screen.blit(text, (20, 20))
            
            pygame.display.flip()
        except pygame.error as e:
            print(f"Display error: {e}")
            self.cleanup()
            sys.exit()

    def train_ai(self):
        try:
            training_states = []
            training_moves = []
            
            for game_num in range(TRAINING_GAMES):
                self.game.reset()
                game_history = []
                
                while not self.game.game_over:
                    # Process events regularly
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.cleanup()
                            sys.exit()
                    
                    state = encode_state(self.game)
                    valid_moves = [i for i in range(9) if self.game.board[i//3][i%3] == '_']
                    
                    if random.random() < 0.2:
                        move = random.choice(valid_moves)
                    else:
                        pred = self.model.predict(state.reshape(1, -1), verbose=0)[0]
                        move = np.argmax([pred[i] if i in valid_moves else -np.inf for i in range(9)])
                    
                    game_history.append((state, move))
                    row, col = decode_move(move)
                    self.game.make_move(row, col)
                    
                    if game_num % 50 == 0:
                        self.draw_board()
                        pygame.time.wait(10)

                # Reward assignment
                reward = 1 if self.game.winner == 'X' else -1 if self.game.winner == 'O' else 0
                for idx, (s, m) in enumerate(game_history):
                    training_states.append(s)
                    training_moves.append(m if reward >= 0 else random.choice(range(9)))

                if game_num % 100 == 0:
                    print(f"Training game {game_num}/{TRAINING_GAMES}")
            
            return np.array(training_states), tf.keras.utils.to_categorical(training_moves, num_classes=9)
        except Exception as e:
            print(f"Training error: {e}")
            self.cleanup()
            sys.exit()

    def cleanup(self):
        pygame.quit()
        tf.keras.backend.clear_session()

    def run(self):
        try:
            # Initial training
            self.training = True
            X, y = self.train_ai()
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=1)
            self.training = False
            
            # Main game loop
            while True:
                self.game.reset()
                game_running = True
                
                while game_running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.cleanup()
                            sys.exit()
                        
                        if not self.game.game_over:
                            if self.game.current_player == 'X':
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    pos = pygame.mouse.get_pos()
                                    col = pos[0] // 200
                                    row = (pos[1] - 100) // 200
                                    if 0 <= row < 3 and 0 <= col < 3:
                                        if self.game.board[row][col] == '_':
                                            self.game.make_move(row, col)
                            else:
                                state = encode_state(self.game)
                                valid_moves = [i for i in range(9) if self.game.board[i//3][i%3] == '_']
                                if valid_moves:
                                    pred = self.model.predict(state.reshape(1, -1), verbose=0)[0]
                                    move = np.argmax([pred[i] if i in valid_moves else -np.inf for i in range(9)])
                                    row, col = decode_move(move)
                                    self.game.make_move(row, col)
                                    pygame.time.wait(300)
                    
                    self.draw_board()
                    clock.tick(30)
                    
                    if self.game.game_over:
                        # Display result
                        result_text = "Draw!" if self.game.winner == 'draw' else f"{self.game.winner} wins!"
                        text = self.font.render(f"{result_text} Click to restart!", True, (255, 255, 255))
                        screen.blit(text, (WIDTH//2 - 120, HEIGHT - 50))
                        pygame.display.flip()
                        
                        # Wait for click
                        waiting = True
                        while waiting:
                            event = pygame.event.wait()
                            if event.type == pygame.QUIT:
                                self.cleanup()
                                sys.exit()
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                waiting = False
                                game_running = False
        except Exception as e:
            print(f"Runtime error: {e}")
            self.cleanup()

if __name__ == "__main__":
    try:
        model = create_model()
        visualizer = GameVisualizer(model)
        visualizer.run()
    except Exception as e:
        print(f"Initialization error: {e}")
        pygame.quit()