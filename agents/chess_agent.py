from utils.chess_utils import check_legal_moves

import numpy as np


class ChessAgent:
    def __init__(self, color):
        self.color = color
        self.king_pos = (0, 4) if self.color == "black" else (7, 4)
        self.pieces = {"pawn": set(), "rook": set(), "knight": set(), "bishop": set(), "queen": set(), "king": set()}
        self.captured_pieces = []
        self.legal_moves = {}
        self.can_castle_kingside = True
        self.can_castle_queenside = True
        self.past_moves = []
    
    def add_piece(self, piece_type, position, board):
        """Add a piece of the given type to the agent's pieces at the given position."""
        self.pieces[piece_type].add(position)
        self.legal_moves[position] = check_legal_moves(board, piece_type, position[0], position[1], self.color)

    def remove_piece(self, piece_type, position):
        self.pieces[piece_type].remove(position)
        del self.legal_moves[position]
    
    def set_king(self, board):
        """Set the position of the king."""
        position = self.king_pos
        self.pieces["king"].add(position)
        self.legal_moves[position] = check_legal_moves(board, "king", position[0], position[1], self.color)
    
    def update_legal_moves(self, board):
        """Update the legal moves for each piece on the board."""
        for piece_type, positions in self.pieces.items():
                for position in positions:
                    self.legal_moves[position] = check_legal_moves(board, piece_type, position[0], position[1], self.color)
                
    def update_relevant_legal_moves(self, board, actions, abbreviations):
        """
        Updates the legal moves for pieces that could have moved from or to the given positions before the move was made.

        Parameters:
        board (2D list): The current state of the chess board
        actions (tuple): A tuple of the form (from_pos, to_pos) representing the move that was made
        abbreviations (dict): A dictionary containing the abbreviations for the chess pieces

        Returns:
        None
        """
        from_pos, to_pos = (actions[0], actions[1]), (actions[2], actions[3])

        for positions, legal_moves in self.legal_moves.items():
            piece = abbreviations[self.color][board[positions[0]][positions[1]]]

            if from_pos in legal_moves or to_pos in legal_moves:
                self.legal_moves[positions] = check_legal_moves(board, piece, positions[0], positions[1], self.color)
    

