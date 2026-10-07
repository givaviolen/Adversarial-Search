import pygame
import sys
import math
import random
import time


PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '
BOARD_SIZE = 5
WIN_CONDITION = 5

# Latihan 3: peluang AI melakukan langkah acak (25%)
RANDOM_MOVE_CHANCE = 0.25

# Penghitung node untuk membandingkan Minimax biasa vs Alpha-Beta (bahan laporan)
node_count = 0


# ===============================================
# BAGIAN 1: LOGIKA PERMAINAN & AI
# ===============================================

def create_board():
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def check_win(board, player):
    # Cek horizontal
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - (WIN_CONDITION - 1)):
            if all(board[r][c+i] == player for i in range(WIN_CONDITION)):
                return True
    # Cek vertikal
    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - (WIN_CONDITION - 1)):
            if all(board[r+i][c] == player for i in range(WIN_CONDITION)):
                return True
    # Cek diagonal (kiri atas ke kanan bawah)
    for r in range(BOARD_SIZE - (WIN_CONDITION - 1)):
        for c in range(BOARD_SIZE - (WIN_CONDITION - 1)):
            if all(board[r+i][c+i] == player for i in range(WIN_CONDITION)):
                return True
    # Cek diagonal (kanan atas ke kiri bawah)
    for r in range(BOARD_SIZE - (WIN_CONDITION - 1)):
        for c in range(WIN_CONDITION - 1, BOARD_SIZE):
            if all(board[r+i][c-i] == player for i in range(WIN_CONDITION)):
                return True
    return False

def is_board_full(board):
    return all(cell != EMPTY for row in board for cell in row)

def get_empty_cells(board):
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == EMPTY]


# ---------- Latihan 2: Evaluation Function yang ditingkatkan ----------

def score_line(line, player):
    """Menilai satu baris/kolom/diagonal dari sudut pandang `player`.
    Semakin banyak simbol berderet (tanpa terhalang), semakin tinggi skornya.
    Ancaman lawan memberi skor negatif."""
    score = 0
    opponent = PLAYER_X if player == PLAYER_O else PLAYER_O
    my_pieces = line.count(player)
    opponent_pieces = line.count(opponent)
    empty_cells = line.count(EMPTY)

    # Peluang milik AI
    if my_pieces == 5: score += 100000
    elif my_pieces == 4 and empty_cells == 1: score += 5000
    elif my_pieces == 3 and empty_cells == 2: score += 100
    elif my_pieces == 2 and empty_cells == 3: score += 10

    # Ancaman milik lawan
    if opponent_pieces == 4 and empty_cells == 1: score -= 4000
    elif opponent_pieces == 3 and empty_cells == 2: score -= 80
    elif opponent_pieces == 2 and empty_cells == 3: score -= 8
    return score

def position_bonus(board):
    """Bonus kecil untuk menguasai tengah papan (tengah ada di lebih banyak jalur menang)."""
    bonus = 0
    mid = BOARD_SIZE // 2
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            weight = (mid - max(abs(r - mid), abs(c - mid))) * 3
            if board[r][c] == PLAYER_O:
                bonus += weight
            elif board[r][c] == PLAYER_X:
                bonus -= weight
    return bonus

def evaluate(board):
    score = 0
    for r in range(BOARD_SIZE):
        row_line = [board[r][c] for c in range(BOARD_SIZE)]
        score += score_line(row_line, PLAYER_O)
    for c in range(BOARD_SIZE):
        col_line = [board[r][c] for r in range(BOARD_SIZE)]
        score += score_line(col_line, PLAYER_O)
    for r in range(BOARD_SIZE - WIN_CONDITION + 1):
        for c in range(BOARD_SIZE - WIN_CONDITION + 1):
            diag1 = [board[r+i][c+i] for i in range(WIN_CONDITION)]
            diag2 = [board[r+i][(c + WIN_CONDITION - 1)-i] for i in range(WIN_CONDITION)]
            score += score_line(diag1, PLAYER_O)
            score += score_line(diag2, PLAYER_O)
    score += position_bonus(board)
    return score


# ---------- Minimax biasa (versi awal, dipakai untuk pembanding) ----------

def minimax(board, depth, is_maximizing, max_depth):
    global node_count
    node_count += 1
    if check_win(board, PLAYER_O): return 100000
    if check_win(board, PLAYER_X): return -100000
    if is_board_full(board): return 0
    if depth == max_depth: return evaluate(board)

    if is_maximizing:
        best_score = -math.inf
        for (row, col) in get_empty_cells(board):
            board[row][col] = PLAYER_O
            score = minimax(board, depth + 1, False, max_depth)
            board[row][col] = EMPTY
            best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for (row, col) in get_empty_cells(board):
            board[row][col] = PLAYER_X
            score = minimax(board, depth + 1, True, max_depth)
            board[row][col] = EMPTY
            best_score = min(best_score, score)
        return best_score


# ---------- Latihan 1: Minimax dengan Alpha-Beta Pruning ----------

def ordered_moves(board):
    """Urutkan langkah dari yang paling dekat tengah papan.
    Langkah bagus dicoba lebih dulu sehingga pemangkasan lebih sering terjadi."""
    mid = BOARD_SIZE // 2
    return sorted(get_empty_cells(board),
                  key=lambda m: max(abs(m[0] - mid), abs(m[1] - mid)))

def minimax_alpha_beta(board, depth, alpha, beta, is_maximizing, max_depth):
    global node_count
    node_count += 1
    # Menang lebih cepat = lebih baik, kalah lebih lambat = lebih baik
    if check_win(board, PLAYER_O): return 100000 - depth
    if check_win(board, PLAYER_X): return -100000 + depth
    if is_board_full(board): return 0
    if depth == max_depth: return evaluate(board)

    if is_maximizing:
        best_score = -math.inf
        for (row, col) in ordered_moves(board):
            board[row][col] = PLAYER_O
            score = minimax_alpha_beta(board, depth + 1, alpha, beta, False, max_depth)
            board[row][col] = EMPTY
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:      # cabang ini tidak akan dipilih MIN -> pangkas
                break
        return best_score
    else:
        best_score = math.inf
        for (row, col) in ordered_moves(board):
            board[row][col] = PLAYER_X
            score = minimax_alpha_beta(board, depth + 1, alpha, beta, True, max_depth)
            board[row][col] = EMPTY
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:      # cabang ini tidak akan dipilih MAX -> pangkas
                break
        return best_score

def get_search_depth(board):
    """Karena Alpha-Beta lebih cepat, AI bisa melihat lebih dalam."""
    return 3 if len(get_empty_cells(board)) > 15 else 4

def find_best_move(board, use_alpha_beta=True, allow_random=True):
    empty_cells = get_empty_cells(board)

    # Latihan 3: 25% kemungkinan melakukan langkah acak
    if allow_random and random.random() < RANDOM_MOVE_CHANCE:
        return random.choice(empty_cells)

    best_score = -math.inf
    best_move = None
    max_depth = get_search_depth(board)

    possible_moves = empty_cells[:]
    random.shuffle(possible_moves)
    mid = BOARD_SIZE // 2
    # sort stabil: urut dekat tengah, tapi tetap acak di antara yang sama jaraknya
    possible_moves.sort(key=lambda m: max(abs(m[0] - mid), abs(m[1] - mid)))

    for (row, col) in possible_moves:
        board[row][col] = PLAYER_O
        if use_alpha_beta:
            move_score = minimax_alpha_beta(board, 0, -math.inf, math.inf, False, max_depth)
        else:
            move_score = minimax(board, 0, False, max_depth)
        board[row][col] = EMPTY
        if move_score > best_score:
            best_score = move_score
            best_move = (row, col)
    return best_move if best_move is not None else random.choice(empty_cells)


# ===============================================
# BAGIAN 2: KODE UI MENGGUNAKAN PYGAME
# ===============================================

pygame.init()

WIDTH, HEIGHT = 600, 700  # Tinggi ditambah untuk status message
SQUARE_SIZE = WIDTH // BOARD_SIZE
RADIUS = SQUARE_SIZE // 2 - 15
LINE_WIDTH = 15
GRID_LINE_COLOR = (20, 170, 156)
BG_COLOR = (28, 190, 175)
PLAYER_X_COLOR = (84, 84, 84)
PLAYER_O_COLOR = (242, 235, 211)
FONT_COLOR = (10, 50, 50)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe 5x5 - AI Minimax + Alpha-Beta")
font = pygame.font.SysFont("Arial", 40, bold=True)

def draw_grid():
    """Menggambar garis grid untuk papan."""
    screen.fill(BG_COLOR)
    for i in range(1, BOARD_SIZE):
        pygame.draw.line(screen, GRID_LINE_COLOR, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, GRID_LINE_COLOR, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, WIDTH), LINE_WIDTH)

def draw_pieces(board):
    """Menggambar X dan O di papan."""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            center_x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = r * SQUARE_SIZE + SQUARE_SIZE // 2

            if board[r][c] == PLAYER_X:
                offset = SQUARE_SIZE // 4
                pygame.draw.line(screen, PLAYER_X_COLOR, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), LINE_WIDTH + 5)
                pygame.draw.line(screen, PLAYER_X_COLOR, (center_x + offset, center_y - offset), (center_x - offset, center_y + offset), LINE_WIDTH + 5)
            elif board[r][c] == PLAYER_O:
                pygame.draw.circle(screen, PLAYER_O_COLOR, (center_x, center_y), RADIUS, LINE_WIDTH)

def draw_status_message(message):
    """Menampilkan pesan status di bagian bawah layar."""
    pygame.draw.rect(screen, BG_COLOR, (0, WIDTH, WIDTH, 100))
    text = font.render(message, True, FONT_COLOR)
    text_rect = text.get_rect(center=(WIDTH // 2, WIDTH + 50))
    screen.blit(text, text_rect)

def main():
    """Loop utama game."""
    board = create_board()
    turn = PLAYER_X
    game_over = False
    message = "Your Turn (X)"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_over:
                    board = create_board()
                    turn = PLAYER_X
                    game_over = False
                    message = "Your Turn (X)"
                    continue

                if turn == PLAYER_X:
                    mouseX = event.pos[0]
                    mouseY = event.pos[1]

                    if mouseY < WIDTH:  # Pastikan klik berada di dalam papan
                        clicked_col = mouseX // SQUARE_SIZE
                        clicked_row = mouseY // SQUARE_SIZE

                        if (clicked_row, clicked_col) in get_empty_cells(board):
                            board[clicked_row][clicked_col] = PLAYER_X

                            if check_win(board, PLAYER_X):
                                message = "You Win! Click to Restart."
                                game_over = True
                            elif is_board_full(board):
                                message = "Draw! Click to Restart."
                                game_over = True
                            else:
                                turn = PLAYER_O

        if turn == PLAYER_O and not game_over:
            draw_grid()
            draw_pieces(board)
            draw_status_message("AI is thinking...")
            pygame.display.update()

            # AI membuat langkah (sambil mencatat waktu & jumlah node untuk laporan)
            global node_count
            node_count = 0
            start = time.time()
            move = find_best_move(board)
            elapsed = time.time() - start
            print(f"[AI] langkah={move} | node dieksplorasi={node_count} | waktu={elapsed:.3f}s")

            if move:
                board[move[0]][move[1]] = PLAYER_O
                if check_win(board, PLAYER_O):
                    message = "AI Wins! Click to Restart."
                    game_over = True
                elif is_board_full(board):
                    message = "Draw! Click to Restart."
                    game_over = True
                else:
                    turn = PLAYER_X
                    message = "Your Turn (X)"

        draw_grid()
        draw_pieces(board)
        draw_status_message(message)
        pygame.display.update()

if __name__ == '__main__':
    main()