import pygame
from pygame import mixer
import sys
import random
import pickle

#Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BOARD_POS_X = (1/15)*SCREEN_WIDTH   #boards relative position with screen
BOARD_POS_Y = (1/15)*SCREEN_HEIGHT
ITEM_POS_X = (1/10)*SCREEN_WIDTH    #the 1st items position w.r.t. screen
ITEM_POS_Y = (1/10)*SCREEN_HEIGHT
ITEM_INC_X = (3/10)*SCREEN_WIDTH    #the relative difference between items
ITEM_INC_Y = (3/10)*SCREEN_HEIGHT
delfault_bg = (255, 255, 255)  # default background color

MUSIC_VOL_MAX = 0.1
SOUND_VOL_MAX = 1

WIDTH, HEIGHT = 600, 600
CELL_SIZE = WIDTH // 3
LINE_WIDTH = 15
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

#Functions
def scale_items(bg1,bg2,bg3,board_image,x_image,x_image_red,o_image,o_image_blue):
    bg1 = pygame.transform.scale(bg1,(SCREEN_WIDTH, SCREEN_HEIGHT))
    bg2 = pygame.transform.scale(bg2,(SCREEN_WIDTH, SCREEN_HEIGHT))
    bg3 = pygame.transform.scale(bg3,(SCREEN_WIDTH, SCREEN_HEIGHT))
    
    board_image = pygame.transform.scale(board_image,(SCREEN_WIDTH*(13/15), SCREEN_HEIGHT*(13/15)))
    x_image = pygame.transform.scale(x_image,(SCREEN_WIDTH*(1/5), SCREEN_HEIGHT*(1/5)))
    x_image_red = pygame.transform.scale(x_image_red,(SCREEN_WIDTH*(1/5), SCREEN_HEIGHT*(1/5)))
    o_image = pygame.transform.scale(o_image,(SCREEN_WIDTH*(1/5), SCREEN_HEIGHT*(1/5)))
    o_image_blue = pygame.transform.scale(o_image_blue,(SCREEN_WIDTH*(1/5), SCREEN_HEIGHT*(1/5)))

    return bg1,bg2,bg3,board_image,x_image,x_image_red,o_image,o_image_blue

def change_background(image_index, background_list):
    new_bg = background_list[image_index]
    image_index = (image_index+2)%(len(background_list))
    return new_bg, image_index

def quit_screen(running,screen):
    quit_prompt = True
    while quit_prompt:
        screen.fill((0, 0, 0))
        show_text(screen,"Are you sure you want to quit? (y/n)",(SCREEN_WIDTH/5, SCREEN_HEIGHT/2.5),40, WHITE)
        pygame.display.update()
        for sub_event in pygame.event.get():
            if sub_event.type == pygame.KEYDOWN:
                if sub_event.key == pygame.K_y:
                    running = False
                    quit_prompt = False
                    pygame.quit()
                    sys.exit()
                if sub_event.key == pygame.K_n:
                    quit_prompt = False
    return running

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
            
        if random.random() < epsilon:   #choose a random possible action; if less than epsilon
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

    def save_q_table(self, filename='models/q_table.pkl'):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename='models/q_table.pkl'):
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
            action = ai.choose_action(state)    #ai makes a choice based on e-greedy method
            if action is None:                  #there is no empty cell remaining i.e. draw
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
        
def human_vs_ai(screen, ai, clock, audio_files, image_files, backgrounds, DISPLAY_BG, IMAGE_INDEX):
    """Human vs AI game loop"""
    # background_list = [bg1, bg2, bg3, DISPLAY_BG, IMAGE_INDEX]
    place_sound = audio_files[0]

    x_image = image_files[1]
    o_image = image_files[3]

    board = [' ']*9
    current_player = 'X'  # AI starts first
    game_over = False
    winner = None

    running = True
    while running:
        #background
        screen.fill(delfault_bg)
        if DISPLAY_BG:
            background = backgrounds[IMAGE_INDEX[0]]
            screen.blit(background, (0,0))
        #board
        screen.blit(image_files[0], (BOARD_POS_X, BOARD_POS_Y))

        #handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            #handle keyboard inputs
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = quit_screen(running,screen)  #call the quit_screen function
                
                if event.key == pygame.K_b:
                    DISPLAY_BG = not DISPLAY_BG
                    background, IMAGE_INDEX[0] = change_background(IMAGE_INDEX[0], backgrounds)

                if event.key == pygame.K_m:
                # change the volume mixer
                    if mixer.music.get_volume() > 0:
                        mixer.music.set_volume(0)
                    elif place_sound.get_volume() > 0:
                        place_sound.set_volume(0)
                    else:
                        mixer.music.set_volume(MUSIC_VOL_MAX)
                        place_sound.set_volume(SOUND_VOL_MAX)
                
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and current_player == 'O':
                place_sound.play()
                x, y = pygame.mouse.get_pos()
                if ((x>BOARD_POS_X and x<BOARD_POS_X+(SCREEN_WIDTH*13/15)) and (y>BOARD_POS_Y and y<BOARD_POS_Y+(SCREEN_HEIGHT*13/15))):
                    x -= BOARD_POS_X
                    y -= BOARD_POS_Y
                    col = x // ((SCREEN_WIDTH*(13/45)))
                    row = y // ((SCREEN_HEIGHT*(13/45)))

                idx = int(row) * 3 + int(col)
                
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
        for idx, cell in enumerate(board):
            if cell == ' ':
                continue
                
            row = idx // 3
            col = idx % 3
            
            if cell == 'X':
                screen.blit(x_image, (ITEM_POS_X + ITEM_INC_X * col , ITEM_POS_Y + ITEM_INC_Y * row))
            else:
                screen.blit(o_image, (ITEM_POS_X + ITEM_INC_X * col , ITEM_POS_Y + ITEM_INC_Y * row))

        # Show game status
        if game_over:
            if winner == 'draw':
                text = "Draw!"
            else:
                text = f"{winner} wins!"
            show_text(screen, text, (SCREEN_WIDTH/2.3, SCREEN_HEIGHT/30), int((1/13)*SCREEN_WIDTH))
            
        pygame.display.flip()
        clock.tick(120) # clock ticks for a constant time/clocks/frames
        if game_over:
            pygame.time.wait(3000)
            screen.fill(delfault_bg)
            if DISPLAY_BG:
                background = backgrounds[IMAGE_INDEX[0]]
                screen.blit(background, (0,0))
            return DISPLAY_BG, IMAGE_INDEX[0]  # return updated values


def main():
        
    #initialize pygame
    pygame.init()

    #initialize the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))   #(width, height)
    clock = pygame.time.Clock()
    
    ai = TicTacToeAI(alpha=0.5, gamma=0.9, epsilon=0.1)
    
    # Try to load existing Q-table
    if not ai.load_q_table():
        print("Training AI...")
        train_ai(screen, ai, num_games=10000)
        ai.save_q_table()
        print("Training complete!")
    
    #HUMAN VS AI
    #import backgrounds and images
    bg1 = pygame.image.load("assets/images/background/bg1.jpg")
    bg2 = pygame.image.load("assets/images/background/bg2.jpg")
    bg3 = pygame.image.load("assets/images/background/bg3.jpg")

    board_image = pygame.image.load("assets/images/board.png")
    x_image = pygame.image.load("assets/images/x.png")
    x_image_red = pygame.image.load("assets/images/x_red.png")
    o_image = pygame.image.load("assets/images/o.png")
    o_image_blue = pygame.image.load("assets/images/o_blue.png")
    line_vert = pygame.image.load("assets/images/line.png")
    icon = pygame.image.load("assets/images/icon.png")

    #Title and Icon
    pygame.display.set_caption("Tic Tac Toe with Q-Learning")
    pygame.display.set_icon(icon)

    #Initialization Error
    if not pygame.font:
        print("Warning, fonts disabled")
    if not pygame.mixer:
        print("Warning, sound disabled")

    #Sounds and Music
    mixer.music.load("assets/audio/music.mp3")
    mixer.music.set_volume(MUSIC_VOL_MAX) # Set the volume in range (0.0 to 1.0)
    mixer.music.play(-1)
    place_sound = mixer.Sound("assets/audio/placing.mp3")
    place_sound.set_volume(SOUND_VOL_MAX)
    
    bg1,bg2,bg3,board_image,x_image,x_image_red,o_image,o_image_blue = scale_items(
    bg1,bg2,bg3,board_image,x_image,x_image_red,o_image,o_image_blue
    )

    #variables
    backgrounds = [bg1, bg2, bg3]
    DISPLAY_BG = False
    IMAGE_INDEX = [0]       #the bg_image index(to choose from various bg)
   
    image_files = (board_image, x_image, x_image_red, o_image, o_image_blue, line_vert)
    audio_files = [place_sound]

    # Main game loop
    while True:
        DISPLAY_BG, IMAGE_INDEX[0] = human_vs_ai(screen, ai,clock, audio_files, image_files, backgrounds, DISPLAY_BG, IMAGE_INDEX)
        
        # Ask to play again
        show_text(screen, "Play again? (Y/N)", (SCREEN_WIDTH//5, SCREEN_HEIGHT//2),int((1/10)*SCREEN_WIDTH))
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
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