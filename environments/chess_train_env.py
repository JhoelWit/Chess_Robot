from typing import Dict
from agents.stockfish_agent import StockfishAgent
from agents.chess_agent import ChessAgent
from utils.chess_utils import check_legal_moves, initialize_board, is_castle, check_in_check, create_partial_mask, can_block_check
from utils.observation_utils import normalize_img

import gym
from gym import spaces
import numpy as np
import pygame
import wandb


class ChessEnv(gym.Env):
    def __init__(self, config):
        self.is_training = config["is_training"]
        self._valid_move_reward = config["valid_move_reward"]
        self._check_reward = config["check_reward"]
        self._win_lose_draw_reward = config["win_lose_draw_reward"]
        self._turn_limit = config["turn_limit"]
        self.stockfish_params = config["stockfish_params"]
        self.level_update_freq = config["level_update_frequency"]
        self.wandb_run = config["wandb_run"]
        self._captured_piece_rewards = {
        "P": 1 * config["pawn_weight"],
        "p": 1 * config["pawn_weight"],
        "R": 5 * config["rook_weight"],
        "r": 5 * config["rook_weight"],
        "N": 3 * config["knight_weight"],
        "n": 3 * config["knight_weight"],
        "B": 3 * config["bishop_weight"],
        "b": 3 * config["bishop_weight"],
        "Q": 9 * config["queen_weight"],
        "q": 9 * config["queen_weight"],
        }
        self._valid_piece_move_rewards = {key: value * 0.5 for key, value in self._captured_piece_rewards.items()}
        self._valid_piece_move_rewards["K"] = 1
        self._valid_piece_move_rewards["k"] = 1

        self._images = { #Images from https://opengameart.org/content/pixel-chess-pieces and https://opengameart.org/content/2d-chess-pack
        "P": pygame.image.load("images/white_pawn.png"),
        "R": pygame.image.load("images/white_rook.png"),
        "N": pygame.image.load("images/white_knight.png"),
        "B": pygame.image.load("images/white_bishop.png"),
        "Q": pygame.image.load("images/white_queen.png"),
        "K": pygame.image.load("images/white_king.png"),
        "p": pygame.image.load("images/black_pawn.png"),
        "r": pygame.image.load("images/black_rook.png"),
        "n": pygame.image.load("images/black_knight.png"),
        "b": pygame.image.load("images/black_bishop.png"),
        "q": pygame.image.load("images/black_queen.png"),
        "k": pygame.image.load("images/black_king.png"),
        }
        self._visual_board_initial = pygame.transform.scale(pygame.image.load("images/wood_board.png"), (config["board_height"], config["board_width"]))
        self._visual_board_initial = pygame.transform.flip(self._visual_board_initial, flip_x=True, flip_y=False)

        self._pieces = {
        "white": {
            "P": "pawn",
            "R": "rook",
            "N": "knight",
            "B": "bishop",
            "Q": "queen",
            "K": "king",
        },
        "black": {
            "p": "pawn",
            "r": "rook",
            "n": "knight",
            "b": "bishop",
            "q": "queen",
            "k": "king",
        },
        }

        #Action and observation space
        self.action_space = spaces.MultiDiscrete([8, 8, 8, 8])
        self.observation_space = spaces.Dict(
            virtual_board=spaces.Box(low=0.0, high=1.0, shape=(3, config["board_height"], config["board_width"]), dtype=np.float32),
            bitboard=spaces.Box(low=0.0, high=1.0, shape=(6, 8, 8), dtype=np.uint8),
            mask = spaces.Box(low=0.0, high=1.0, shape=(32,), dtype=np.uint8)
        )
        # Global counters (not to be reset)
        self.global_step = 1
        self.global_episode = 1

    def step(self, actions):
        white_piece_moved = self.board[actions[0]][actions[1]]
        white_valid_move, white_captured_piece, white_in_check, white_promoted = self.play(actions)
        if white_valid_move and not white_in_check:
            self.update_stockfish_state(actions, white_promoted)
            stockfish_move = self.black_agent.return_best_move()
            black_valid_move, black_captured_piece, black_in_check, black_promoted = self.play(stockfish_move)
            self.update_stockfish_state(stockfish_move, black_promoted)
        else:
            black_valid_move, black_captured_piece, black_in_check, black_promoted = self.play(None)

        reward = self.calculate_reward(white_piece_moved, white_valid_move, white_captured_piece, white_in_check, black_valid_move, black_captured_piece, black_in_check)
        observation = self.state_space_generation()
        info = self.load_info()
        done = self.check_if_done() if (white_valid_move and black_valid_move) else True
        if not self.is_training and self.wandb_run and done:
            self.render_episode()
        self.update_counter()
        return observation, reward, done, info
    
    def update_stockfish_state(self, move, promoted=None):
        self.switch_player()
        try:
            self.black_agent.update_game_state(move, promoted) if type(move) == np.ndarray else None
        except ValueError as e:
            self.print_debug_message(e, move)


    def reset(self):
        self.board = initialize_board()
        self.virtual_states = []
        self.turn_counter = 1
        self.white_agent, self.black_agent = self.create_agents(color="white", board=self.board, stockfish_params=self.stockfish_params)
        self.current_player = "white"
        observation = self.state_space_generation()
        #curriculum learning - levels max out at 20 for stockfish
        if self.is_training:
            if self.global_step % self.level_update_freq == 0:
                self.adversary_level = (self.stockfish_params["engine_params"]["Skill Level"] + 1) % 16
                self.black_agent.update_skill_level(self.adversary_level)
                self.stockfish_params["engine_params"]["Skill Level"] = self.adversary_level
        else:
            self.adversary_level = np.random.randint(20)
            self.black_agent.update_skill_level(self.adversary_level)
        self.global_episode += 1
        return observation

    def close(self):
        pass

    def render(self):
        pass

    def render_episode(self):
        for state in self.virtual_states:
            self.wandb_run.log({f"episode_{self.global_episode}_no._turns_{self.turn_counter}": wandb.Image(state)})


    def play(self, actions):
        """
        Apply the given actions to the environment.

        Parameters:
        actions (np.ndarray): The actions to be applied to the environment, represented as a 4-tuple of integers (from_row, from_col, to_row, to_col)

        Returns:
        tuple: A 4-tuple containing the following information:
            - success (bool): Indicates whether the actions were successfully applied to the environment
            - captured_piece (str or None): The piece that was captured, if any. Returns None if no piece was captured.
            - in_check (bool): Indicates whether the current player's king is in check after the move
            - is_piece_promoted (bool): Indicates whether a pawn was promoted as a result of the move

        """

        if not type(actions) == np.ndarray:
            return False, None, False, False

        # Validate the actions
        from_row, from_col, to_row, to_col = actions

        friendly_pieces = {"P", "R", "N", "B", "Q", "K"} if self.current_player == "white" else {"p", "r", "n", "b", "q", "k"}

        if self.board[from_row][from_col] == "." or self.board[to_row][to_col] in friendly_pieces or self.board[from_row][from_col] not in friendly_pieces:
            return False, None, False, False

        current_agent = self.white_agent if self.current_player == "white" else self.black_agent
        opposing_agent = self.black_agent if self.current_player == "white" else self.white_agent
        chosen_piece = self._pieces[self.current_player][self.board[from_row][from_col]]
        legal_moves = check_legal_moves(self.board, chosen_piece, from_row, from_col, self.current_player)

        # Check if the chosen move is a legal move
        if (to_row, to_col) in legal_moves:
            self.board, captured_piece, is_piece_promoted = self.update_board(self.board, actions)
            king_pos = next(iter(current_agent.pieces["king"]))
            if check_in_check(self.board, opposing_agent, king_pos[0], king_pos[1], self._pieces):
                return False, captured_piece, True, is_piece_promoted
            return True, captured_piece, False, is_piece_promoted

        # Check if the chosen move is a valid castle
        elif is_castle(from_row, from_col, to_row, to_col, current_agent):
            self.board, no_captured_piece, is_piece_promoted = self.update_board(self.board, actions, is_castle=True)
            king_pos = next(iter(current_agent.pieces["king"]))
            if check_in_check(self.board, opposing_agent, king_pos[0], king_pos[1], self._pieces):
                return False, captured_piece, True, is_piece_promoted
            return True, no_captured_piece, False, is_piece_promoted

        # Return early if the move isn't valid
        else:
            return False, None, False, False

    def calculate_reward(self, white_piece_moved, white_valid_move, white_captured_piece, white_in_check, black_valid_move, black_captured_piece, black_in_check):
        reward = 0
        # Reward for making a valid move or penalty for checkmate
        if white_valid_move and not white_in_check:
            reward += self._valid_piece_move_rewards[white_piece_moved]
        elif white_in_check:
            reward -= 10
            return reward
        elif not white_valid_move:
            reward -= self._valid_move_reward
        
        # Reward for capturing a piece or favorable trades
        if white_captured_piece and black_captured_piece:
            reward += self._captured_piece_rewards[white_captured_piece] - self._captured_piece_rewards[black_captured_piece]
        elif white_captured_piece:
            reward += self._captured_piece_rewards[white_captured_piece]
        elif black_captured_piece:
            reward -= self._captured_piece_rewards[black_captured_piece]

        # Reward for checking the adversary king
        if black_in_check: #TODO(jwitter) I'll have to actually fix this modifier to see if black is in check after white moves, and not after black moves.
            reward += self._check_reward

        #TODO(jwitter) Implement rewards for stale moves (i.e agent moves same piece twice in a row or repeats the same sequence) to promote development.

        return reward

    def state_space_generation(self):
        obs = {}
        obs["virtual_board"] = normalize_img(self.draw_board())
        obs["bitboard"] = self.get_bitboard()
        obs["mask"] = create_partial_mask(self.board, self.current_player)
        return obs
        
    def check_if_done(self):
        #TODO(jwitter) add stale-mate checks.
        if self.turn_counter >= self._turn_limit or self.is_checkmate():
            print(f"Maximum number of turns have passed." if self.turn_counter >= self._turn_limit else f"Checkmate has occurred")
            return True
        else:
            return False

    def update_counter(self):
        self.turn_counter += 1
        self.global_step += 1
    
    def switch_player(self):
        self.current_player = "white" if self.current_player == "black" else "black"

    @staticmethod
    def create_agents(color=None, board=None, stockfish_params=None, moveset_test=False):
        """Create the white and black agents and add the starting pieces."""
        def add_starting_pieces(agent: ChessAgent, rows):
            """Add the starting pieces for the given agent."""
            agent.add_piece("rook", rows[0], board)
            agent.add_piece("rook", rows[7], board)
            agent.add_piece("knight", rows[1], board)
            agent.add_piece("knight", rows[6], board)
            agent.add_piece("bishop", rows[2], board)
            agent.add_piece("bishop", rows[5], board)
            agent.add_piece("queen", rows[3], board)

        if color is None:
            white_agent = StockfishAgent("white", stockfish_params) if not moveset_test else ChessAgent("white")
            black_agent = StockfishAgent("black", stockfish_params) if not moveset_test else ChessAgent("black")
            add_starting_pieces(white_agent, [(7, col) for col in range(8)])
            add_starting_pieces(black_agent, [(0, col) for col in range(8)])
            for col in range(8):
                white_agent.add_piece("pawn", (6, col), board)
                black_agent.add_piece("pawn", (1, col), board)
            # Set the positions of the kings
            white_agent.set_king(board)
            black_agent.set_king(board)
            return white_agent, black_agent
        else:
            agent = ChessAgent(color)
            adversary = StockfishAgent("black" if color=="white" else "white", stockfish_params)
            agent_rows = [(7, col) for col in range(8)] if color == "white" else [(0, col) for col in range(8)]
            adversary_rows = [(7, col) for col in range(8)] if color == "black" else [(0, col) for col in range(8)]
            add_starting_pieces(agent, agent_rows)
            add_starting_pieces(adversary, adversary_rows)
            # Add pawns to the agent's pieces
            for col in range(8):
                agent.add_piece("pawn", (6, col), board) if color == "white" else agent.add_piece("pawn", (1, col), board)
                adversary.add_piece("pawn", (6, col), board) if color == "black" else adversary.add_piece("pawn", (1, col), board)
            # Set the position of the king
            agent.set_king(board)
            adversary.set_king(board)
            return agent, adversary

    def update_board(self, board, actions, is_castle=False):
        from_row, from_col, to_row, to_col = actions
        agent = self.white_agent if self.current_player == "white" else self.black_agent
        opposing_agent = self.black_agent if self.current_player == "white" else self.white_agent
        agent.past_moves.append((from_row, from_col)) #Used to help with castling
        is_piece_promoted = False

        if is_castle:
            # Perform castle move
            if to_col > from_col:
                # Kingside castle
                board[from_row][from_col+2] = board[from_row][from_col]
                board[from_row][from_col] = "."
                board[from_row][from_col+1] = board[from_row][from_col+3]
                board[from_row][from_col+3] = "."

                #Update agent pieces
                agent.remove_piece("rook", (from_row, from_col + 3))
                agent.add_piece("rook", (from_row, from_col + 1), board)

                agent.remove_piece("king", (from_row, from_col))
                agent.add_piece("king", (from_row, from_col + 2), board)

            else:
                # Queenside castle
                board[from_row][from_col-2] = board[from_row][from_col]
                board[from_row][from_col] = "."
                board[from_row][from_col-1] = board[from_row][from_col-4]
                board[from_row][from_col-4] = "."

                #Update agent pieces
                agent.remove_piece("rook", (from_row, from_col - 4))
                agent.add_piece("rook", (from_row, from_col - 1), board)

                agent.remove_piece("king", (from_row, from_col))
                agent.add_piece("king", (from_row, from_col - 2), board)
            return board, None, is_piece_promoted
        
        # Get the piece at the from position
        piece = board[from_row][from_col]

        # Handle pawn promotion (if any)
        if (piece == "p" and to_row == 7) or (piece == "P" and to_row == 0):
            promoted_piece = "Q" if piece == "P" else "q"
            agent.remove_piece(self._pieces[agent.color][piece], (from_row, from_col))
            agent.add_piece(self._pieces[agent.color][promoted_piece], (to_row, to_col), board)
            piece = promoted_piece
            is_piece_promoted = True
        else:
            agent.remove_piece(self._pieces[agent.color][piece], (from_row, from_col))
            agent.add_piece(self._pieces[agent.color][piece], (to_row, to_col), board)

        # Check if a piece was captured
        captured_piece = board[to_row][to_col] if board[to_row][to_col] != "." else None

        if captured_piece:
            agent.captured_pieces.append(captured_piece)
            # Update opposing agent with removed piece
            opposing_agent.remove_piece(self._pieces[opposing_agent.color][captured_piece], (to_row, to_col))

        # Update the board with the moved piece
        board[to_row][to_col] = piece
        
        # Clear the from position on the board
        board[from_row][from_col] = "."

        # Update the necessary legal moves for each agent
        # agent.update_relevant_legal_moves(board, actions, self._pieces) #TODO(jwitter) This method doesn't work unless friendly fire is on (joke)
        # opposing_agent.update_relevant_legal_moves(board, actions, self._pieces)
        agent.update_legal_moves(board)
        opposing_agent.update_legal_moves(board)

        self.check_castle_conditions(board)
        
        return board, captured_piece, is_piece_promoted

    def get_bitboard(self):
        # Initialize all bitboards as empty
        pawn_bitboard = np.zeros((8, 8))
        rook_bitboard = np.zeros((8, 8))
        knight_bitboard = np.zeros((8, 8))
        bishop_bitboard = np.zeros((8, 8))
        queen_bitboard = np.zeros((8, 8))
        king_bitboard = np.zeros((8, 8))

        # Iterate through the board and set the corresponding positions in the bitboards
        for i in range(8):
            for j in range(8):
                if self.board[i][j] != ".":
                    if self.board[i][j] == "P" if self.current_player == "white" else "p":
                        pawn_bitboard[i][j] = 1
                    elif self.board[i][j] == "R" if self.current_player == "white" else "r":
                        rook_bitboard[i][j] = 1
                    elif self.board[i][j] == "N" if self.current_player == "white" else "n":
                        knight_bitboard[i][j] = 1
                    elif self.board[i][j] == "B" if self.current_player == "white" else "b":
                        bishop_bitboard[i][j] = 1
                    elif self.board[i][j] == "Q" if self.current_player == "white" else "q":
                        queen_bitboard[i][j] = 1
                    elif self.board[i][j] == "K" if self.current_player == "white" else "k":
                        king_bitboard[i][j] = 1

        # Stack the bitboards together and return the result
        return np.stack((pawn_bitboard, rook_bitboard, knight_bitboard, bishop_bitboard, queen_bitboard, king_bitboard))

    def draw_board(self):
        # Get the size of the squares on the chess board
        board = self._visual_board_initial.copy()

        square_width = board.get_width() // len(self.board[0])
        square_height = board.get_height() // len(self.board)

        for i, row in enumerate(self.board):
            for j, piece in enumerate(row):
                if piece == ".":
                    continue
                x = j * square_width
                y = i * square_height

                piece_image = pygame.transform.scale(self._images[piece], (square_height, square_width)) #TODO(jwitter) do this in initialization, we're gonna need to speed up where ever.
                board.blit(piece_image, (x, y))

        board = pygame.transform.flip(board, flip_x=True, flip_y=False)
        self.current_virtual_state = np.rot90(pygame.surfarray.array3d(board), k=-1)
        self.virtual_states.append(self.current_virtual_state)

        return self.current_virtual_state.reshape(3, board.get_height(), board.get_width())

    def check_castle_conditions(self, board):
        
        castle_positions = {
            "kingside": [(7, 5), (7, 6)] if self.current_player == "white" else [(0, 5), (0, 6)],
            "queenside": [(7, 3), (7, 2), (7, 1)] if self.current_player == "white" else [(0, 3), (0, 2), (0, 1)]
        }
        king_row, king_col = (7, 4) if self.current_player == "white" else (0, 4)
        current_agent = self.white_agent if self.current_player == "white" else self.black_agent
        opposing_agent = self.black_agent if self.current_player == "white" else self.white_agent

        # Initialize as true
        current_agent.can_castle_kingside = True
        current_agent.can_castle_queenside = True

        # Check if the king has moved
        if (king_row, king_col) in current_agent.past_moves:
            current_agent.can_castle_kingside = False
            current_agent.can_castle_queenside = False
            return

        # Check if the kingside rook has moved
        if (king_row, 7) in current_agent.past_moves:
            current_agent.can_castle_kingside = False

        # Check if the queenside rook has moved
        if (king_row, 0) in current_agent.past_moves:
            current_agent.can_castle_queenside = False

        # Check if the kingside castle lane is blocked
        blocked_positions = [board[pos[0], pos[1]] for pos in castle_positions["kingside"]]
        kingside_blocked = np.any([pos != '.' for pos in blocked_positions])
        if current_agent.can_castle_kingside and (kingside_blocked or any(check_in_check(board, opposing_agent, pos[0], pos[1], self._pieces) for pos in castle_positions["kingside"])):
            current_agent.can_castle_kingside = False

        # Check if the queenside castle lane is blocked
        blocked_positions = [board[pos[0], pos[1]] for pos in castle_positions["queenside"]]
        queenside_blocked = np.any([pos != '.' for pos in blocked_positions])
        if current_agent.can_castle_queenside and (queenside_blocked or any(check_in_check(board, opposing_agent, pos[0], pos[1], self._pieces) for pos in castle_positions["queenside"])):
            current_agent.can_castle_queenside = False

    def is_checkmate(self):
        for agent in [self.white_agent, self.black_agent]:
            opposing_agent = self.white_agent if agent.color == "black" else self.black_agent
            king_pos = next(iter(agent.pieces["king"]))
            info = check_in_check(self.board, opposing_agent, king_pos[0], king_pos[1], self._pieces, count_checks=True)
            if not isinstance(info, Dict):
                continue
            elif info["number_of_checks"] >= 2: #Double check, king must move
                king_moves = [check_in_check(self.board, opposing_agent, pos[0], pos[1], self._pieces) for pos in agent.legal_moves[king_pos]]
                if len(agent.legal_moves[king_pos]) >= 1 and any(not x for x in king_moves):
                    return False
                else:
                    return True
            else: #Single check, king can move or team piece can support
                attacking_pos = info["attacker_positions"][0]
                king_moves = [check_in_check(self.board, opposing_agent, pos[0], pos[1], self._pieces) for pos in agent.legal_moves[king_pos]]
                can_capture_attacker = [attacking_pos in legal_moves_list for legal_moves_list in agent.legal_moves.values()]

                if ((len(agent.legal_moves[king_pos]) >= 1 and any(x == False for x in king_moves)) or any(can_capture_attacker) 
                or can_block_check(attacking_pos, king_pos, agent.legal_moves)):
                    return False
                else:
                    return True
        return False

    def load_info(self):
        info = {}
        if self.is_training:
            return info
        else:
            info["game_state"] = self.current_virtual_state
            return info

    def print_debug_message(self, error, actions):
        print(f"{error} \n {actions} \n {self.black_agent.engine.get_board_visual()}")

    def set_training_mode(self, mode):
        self.is_training = mode
    
    def set_wandb_run(self, run):
        self.wandb_run = run
