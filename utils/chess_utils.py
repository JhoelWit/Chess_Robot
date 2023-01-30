import itertools

import numpy as np


def check_legal_moves(board, piece_type, row, col, player_color):
    legal_moves = set()
    enemy_pieces = {"p", "r", "n", "b", "q", "k"} if player_color == "white" else {"P", "R", "N", "B", "Q", "K"}

    # Common functions we can re-use

    def check_rook_moves():
        # Rooks can move horizontally or vertically
        for i in range(row - 1, -1, -1):
            if board[i][col] == ".":
                legal_moves.add((i, col))
            elif board[i][col] in enemy_pieces:
                legal_moves.add((i, col))
                break
            else:
                break
        for i in range(row + 1, 8):
            if board[i][col] == ".":
                legal_moves.add((i, col))
            elif board[i][col] in enemy_pieces:
                legal_moves.add((i, col))
                break
            else:
                break
        for j in range(col - 1, -1, -1):
            if board[row][j] == ".":
                legal_moves.add((row, j))
            elif board[row][j] in enemy_pieces:
                legal_moves.add((row, j))
                break
            else:
                break
        for j in range(col + 1, 8):
            if board[row][j] == ".":
                legal_moves.add((row, j))
            elif board[row][j] in enemy_pieces:
                legal_moves.add((row, j))
                break
            else:
                break

    def check_bishop_moves():
        # Bishops can move diagonally
        for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
            if board[i][j] == ".":
                legal_moves.add((i, j))
            elif board[i][j] in enemy_pieces:
                legal_moves.add((i, j))
                break
            else:
                break
        for i, j in zip(range(row - 1, -1, -1), range(col + 1, 8)):
            if board[i][j] == ".":
                legal_moves.add((i, j))
            elif board[i][j] in enemy_pieces:
                legal_moves.add((i, j))
                break
            else:
                break
        for i, j in zip(range(row + 1, 8), range(col - 1, -1, -1)):
            if board[i][j] == ".":
                legal_moves.add((i, j))
            elif board[i][j] in enemy_pieces:
                legal_moves.add((i, j))
                break
            else:
                break
        for i, j in zip(range(row + 1, 8), range(col + 1, 8)):
            if board[i][j] == ".":
                legal_moves.add((i, j))
            elif board[i][j] in enemy_pieces:
                legal_moves.add((i, j))
                break
            else:
                break
    
    if piece_type == "pawn":
        # Pawns can move forward one space, or capture diagonally
        if player_color == "white":
            if row > 0:
                if board[row - 1][col] == ".":
                    legal_moves.add((row - 1, col))
                    if row == 6 and board[row - 2][col] == ".":
                        legal_moves.add((row - 2, col))
                if col > 0 and board[row - 1][col - 1] in enemy_pieces:
                    legal_moves.add((row - 1, col - 1))
                if col < 7 and board[row - 1][col + 1] in enemy_pieces:
                    legal_moves.add((row - 1, col + 1))
        elif player_color == "black":
            if row < 7:
                if board[row + 1][col] == ".":
                    legal_moves.add((row + 1, col))
                    if row == 1 and board[row + 2][col] == ".":
                        legal_moves.add((row + 2, col))
                if col > 0 and board[row + 1][col - 1] in enemy_pieces:
                    legal_moves.add((row + 1, col - 1))
                if col < 7 and board[row + 1][col + 1] in enemy_pieces:
                    legal_moves.add((row + 1, col + 1))

    elif piece_type == "rook":
        check_rook_moves()
    
    elif piece_type == "bishop":
        check_bishop_moves()

    elif piece_type == "queen":
        # Queens can move horizontally, vertically, or diagonally
        check_rook_moves()
        check_bishop_moves()

    elif piece_type == "knight":
        # Knights can move in an "L" shape
        if row > 1:
            if col > 0 and (board[row - 2][col - 1] == "." or board[row - 2][col - 1] in enemy_pieces):
                legal_moves.add((row - 2, col - 1))
            if col < 7 and (board[row - 2][col + 1] == "." or board[row - 2][col + 1] in enemy_pieces):
                legal_moves.add((row - 2, col + 1))
        if row > 0:
            if col > 1 and (board[row - 1][col - 2] == "." or board[row - 1][col - 2] in enemy_pieces):
                legal_moves.add((row - 1, col - 2))
            if col < 6 and (board[row - 1][col + 2] == "." or board[row - 1][col + 2] in enemy_pieces):
                legal_moves.add((row - 1, col + 2))
        if row < 7:
            if col > 1 and (board[row + 1][col - 2] == "." or board[row + 1][col - 2] in enemy_pieces):
                legal_moves.add((row + 1, col - 2))
            if col < 6 and (board[row + 1][col + 2] == "." or board[row + 1][col + 2] in enemy_pieces):
                legal_moves.add((row + 1, col + 2))
        if row < 6:
            if col > 0 and (board[row + 2][col - 1] == "." or board[row + 2][col - 1] in enemy_pieces):
                legal_moves.add((row + 2, col - 1))
            if col < 7 and (board[row + 2][col + 1] == "." or board[row + 2][col + 1] in enemy_pieces):
                legal_moves.add((row + 2, col + 1))

    elif piece_type == "king":
        # Kings can move one space in any direction
        if row > 0:
            if (board[row - 1][col] == "." or board[row - 1][col] in enemy_pieces):
                legal_moves.add((row - 1, col))
            if col > 0 and (board[row - 1][col - 1] == "." or board[row - 1][col - 1] in enemy_pieces):
                legal_moves.add((row - 1, col - 1))
            if col < 7 and (board[row - 1][col + 1] == "." or board[row - 1][col + 1] in enemy_pieces):
                legal_moves.add((row - 1, col + 1))
        if row < 7:
            if (board[row + 1][col] == "." or board[row + 1][col] in enemy_pieces):
                legal_moves.add((row + 1, col))
            if col > 0 and (board[row + 1][col - 1] == "." or board[row + 1][col - 1] in enemy_pieces):
                legal_moves.add((row + 1, col - 1))
            if col < 7 and (board[row + 1][col + 1] == "." or board[row + 1][col + 1] in enemy_pieces):
                legal_moves.add((row + 1, col + 1))
        if col > 0 and (board[row][col - 1] == "." or board[row][col - 1] in enemy_pieces):
            legal_moves.add((row, col - 1))
        if col < 7 and (board[row][col + 1] == "." or board[row][col + 1] in enemy_pieces):
            legal_moves.add((row, col + 1))

    return legal_moves

def initialize_board():
    board = np.array([
    ["r", "n", "b", "q", "k", "b", "n", "r"],# 8 0
    ["p", "p", "p", "p", "p", "p", "p", "p"],# 7 1
    [".", ".", ".", ".", ".", ".", ".", "."],# 6 2
    [".", ".", ".", ".", ".", ".", ".", "."],# 5 3
    [".", ".", ".", ".", ".", ".", ".", "."],# 4 4
    [".", ".", ".", ".", ".", ".", ".", "."],# 3 5
    ["P", "P", "P", "P", "P", "P", "P", "P"],# 2 6
    ["R", "N", "B", "Q", "K", "B", "N", "R"],# 1 7
    ]#A    B    C    D    E    F    G    H
    )#0    1    2    3    4    5    6    7

    return board

def is_castle(from_row, from_col, to_row, to_col, agent):
    """Check if a move is a valid castle."""
    king_initial_pos = agent.king_pos
    king_side_pos = (7, 6) if agent.color == "white" else (0, 6)
    queen_side_pos = (7, 2) if agent.color == "white" else (0, 2)

    # Check if the move is a kingside castle    
    if (from_row, from_col) == king_initial_pos and (to_row, to_col) == king_side_pos:
        return agent.can_castle_kingside
    # Check if the move is a queenside castle
    elif (from_row, from_col) == king_initial_pos and (to_row, to_col) == queen_side_pos:
        return agent.can_castle_queenside
    # Return False if the move is not a castle
    return False

def check_in_check(board, opposing_agent, row, col, abbreviations, count_checks = False):
    checks, attackers = 0, []
    for pos, legal_moves in opposing_agent.legal_moves.items():
        opposing_piece = abbreviations[opposing_agent.color][board[pos[0]][pos[1]]]
        if opposing_piece == "pawn":
            # Pawns can only attack diagonally
            if abs(row - pos[0]) == 1 and abs(col - pos[1]) == 1:
                # Check if the pawn is attacking the correct color
                if (opposing_agent.color == "white" and row < pos[0]) or (opposing_agent.color == "black" and row > pos[0]):
                    if count_checks:
                        checks += 1
                        attackers.append(pos)
                    else:
                        # print(f" {row, col} king in check because pawn at {pos} can hurt it")
                        return True
        elif (row, col) in legal_moves:
            if count_checks:
                checks += 1
                attackers.append(pos)
            else:
                # print(f" {row, col} king in check because {opposing_piece} at {pos} can hurt it")
                return True

    return False if (not count_checks or checks == 0) else {"number_of_checks":checks, "attacker_positions":attackers}

def can_block_check(attacking_pos, king_pos, legal_moves_dict):
    """
    Check if a friendly piece can block the check on the king.
    attacking_pos: tuple of ints (row, col) representing the position of the attacking piece
    king_pos: tuple of ints (row, col) representing the position of the king
    legal_moves_dict: dict of strings (friendly pieces) and lists of tuples (their legal moves)
    """
    flattened_moves = list(itertools.chain.from_iterable(legal_moves_dict.values()))
    direction = (king_pos[0] - attacking_pos[0], king_pos[1] - attacking_pos[1])
    direction = tuple(map(lambda x: x // abs(x) if x != 0 else 0, direction))
    curr_pos = (attacking_pos[0] + direction[0], attacking_pos[1] + direction[1])
    moves = 1 # Failsafe
    while curr_pos != king_pos and moves < 8:
        if curr_pos in flattened_moves:
            return True
        curr_pos = (curr_pos[0] + direction[0], curr_pos[1] + direction[1])
        moves += 1
    return False

def create_partial_mask(board, color):
    friendly_pieces = ["P", "R", "N", "B", "Q", "K"] if color == "white" else ["p", "r", "n", "b", "q", "k"] # Sets don't work with numpy isin... found out the hard way...

    mask = np.ones(32)
    # Check if all elements in each row are in the friendly_pieces list
    row_mask = np.any(np.isin(board, friendly_pieces), axis=1)
    # Check if all elements in each column are in the friendly_pieces list
    col_mask = np.any(np.isin(board, friendly_pieces), axis=0)
    # Create a 1D numpy array of the mask
    mask[:16] = np.concatenate((row_mask, col_mask))
    return mask
