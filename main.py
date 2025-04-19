import pygame
import sys
import random
import pickle

# Game constants
WIDTH, HEIGHT = 600, 600
CELL_SIZE = WIDTH // 3
LINE_WIDTH = 15
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def check_winner(board, player):
    """Check if the specified player has won"""
    # Rows
    for i in range(0, 9, 3):
        if board[i] == board[i+1] == board[i+2] == player:
            return True
    # Columns
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] == player:
            return True
    # Diagonals
    if board[0] == board[4] == board[8] == player:
        return True
    if board[2] == board[4] == board[6] == player:
        return True
    return False

def is_board_full(board):
    """Check if the board is completely filled"""
    return ' ' not in board

class TicTacToeAI:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_possible_actions(self, state):
        """Return list of available moves (indices)"""
        return [i for i, cell in enumerate(state) if cell == ' ']

    def choose_action(self, state, epsilon=None):
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon
            
        possible_actions = self.get_possible_actions(state)
        if not possible_actions:
            return None
            
        if random.random() < epsilon:
            return random.choice(possible_actions)
        else:
            # Choose best action based on Q-values
            q_values = [self.q_table.get((state, a), 0) for a in possible_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q_table(self, history, winner):
        """Update Q-values based on game history and outcome"""
        rewards = {'X': 0, 'O': 0, 'draw': 0}
        if winner == 'X':
            rewards = {'X': 1, 'O': -1}
        elif winner == 'O':
            rewards = {'X': -1, 'O': 1}
            
        for i, (state, action, player) in enumerate(history):
            steps_remaining = len(history) - i - 1
            reward = rewards[player] * (self.gamma ** steps_remaining)
            old_q = self.q_table.get((state, action), 0)
            self.q_table[(state, action)] = old_q + self.alpha * (reward - old_q)

    def save_q_table(self, filename='q_table.pkl'):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename='q_table.pkl'):
        """Load Q-table from file"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

def draw_board(screen, board):
    """Draw the game board and pieces"""
    screen.fill(WHITE)
    
    # Draw grid lines
    for i in range(1, 3):
        pygame.draw.line(screen, BLACK, (i*CELL_SIZE, 0), (i*CELL_SIZE, HEIGHT), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (0, i*CELL_SIZE), (WIDTH, i*CELL_SIZE), LINE_WIDTH)
    
    # Draw X's and O's
    for idx, cell in enumerate(board):
        if cell == ' ':
            continue
            
        row = idx // 3
        col = idx % 3
        x = col * CELL_SIZE + CELL_SIZE//2
        y = row * CELL_SIZE + CELL_SIZE//2
        
        if cell == 'X':
            color = RED
            pygame.draw.line(screen, color, (x-50, y-50), (x+50, y+50), 5)
            pygame.draw.line(screen, color, (x+50, y-50), (x-50, y+50), 5)
        else:
            color = BLUE
            pygame.draw.circle(screen, color, (x, y), 50, 5)

def show_text(screen, text, pos, size=30, color=BLACK):
    """Utility function to display text"""
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)

def train_ai(screen, ai, num_games=10000):
    """Train the AI through self-play with visualization"""
    for game in range(num_games):
        board = [' ']*9
        current_player = 'X'
        history = []
        winner = None
        
        while True:
            state = tuple(board)
            action = ai.choose_action(state)
            if action is None:
                winner = 'draw'
                break
                
            # Make move
            new_board = list(board)
            new_board[action] = current_player
            new_board = tuple(new_board)
            history.append((state, action, current_player))
            
            # Check game status
            if check_winner(new_board, current_player):
                winner = current_player
                break
            elif is_board_full(new_board):
                winner = 'draw'
                break
                
            board = new_board
            current_player = 'O' if current_player == 'X' else 'X'
            
            # Update display
            draw_board(screen, new_board)
            show_text(screen, f"Training Game: {game+1}/{num_games}", (10, 10))
            pygame.display.flip()
            pygame.time.wait(50)
            
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        
        ai.update_q_table(history, winner)
        
def human_vs_ai(screen, ai):
    """Human vs AI game loop"""
    board = [' ']*9
    current_player = 'X'  # AI starts first
    game_over = False
    winner = None
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and current_player == 'O':
                x, y = pygame.mouse.get_pos()
                col = x // CELL_SIZE
                row = y // CELL_SIZE
                idx = row * 3 + col
                
                if board[idx] == ' ':
                    board[idx] = 'O'
                    
                    if check_winner(board, 'O'):
                        winner = 'O'
                        game_over = True
                    elif is_board_full(board):
                        winner = 'draw'
                        game_over = True
                    else:
                        current_player = 'X'
        
        if not game_over and current_player == 'X':
            # AI's turn
            state = tuple(board)
            action = ai.choose_action(state, epsilon=0)
            if action is not None:
                board[action] = 'X'
                
                if check_winner(board, 'X'):
                    winner = 'X'
                    game_over = True
                elif is_board_full(board):
                    winner = 'draw'
                    game_over = True
                else:
                    current_player = 'O'
        
        # Draw board
        draw_board(screen, board)
        
        # Show game status
        if game_over:
            if winner == 'draw':
                text = "Draw!"
            else:
                text = f"{winner} wins!"
            show_text(screen, text, (WIDTH//2-100, HEIGHT//2-20), 74)
            
        pygame.display.flip()
        
        if game_over:
            pygame.time.wait(3000)
            break

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tic Tac Toe with Q-Learning")
    clock = pygame.time.Clock()
    
    ai = TicTacToeAI(alpha=0.5, gamma=0.9, epsilon=0.1)
    
    # Try to load existing Q-table
    if not ai.load_q_table():
        print("Training AI...")
        train_ai(screen, ai, num_games=10000)
        ai.save_q_table()
        print("Training complete!")
    
    # Main game loop
    while True:
        human_vs_ai(screen, ai)
        
        # Ask to play again
        screen.fill(WHITE)
        show_text(screen, "Play again? (Y/N)", (WIDTH//2-100, HEIGHT//2))
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        waiting = False
                    elif event.key == pygame.K_n:
                        pygame.quit()
                        sys.exit()

if __name__ == "__main__":
    main()