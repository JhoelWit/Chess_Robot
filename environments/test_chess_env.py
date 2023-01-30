from environments.chess_train_env import ChessEnv
from utils.chess_utils import initialize_board

import numpy as np


class TestChessEnv(ChessEnv):
    def __init__(self, config):
        super().__init__(config)
        self.is_training = False
        self._moveset_testing = False

    def step_stockfish(self, actions):
        # White to move
        white_stockfish_move = self.white_agent.return_best_move()
        white_piece_moved = self.board[white_stockfish_move[0]][white_stockfish_move[1]] if type(white_stockfish_move) == np.ndarray else None
        white_valid_move, white_captured_piece, white_in_check, white_promoted = self.play(white_stockfish_move)
        if white_valid_move and not white_in_check:
            self.update_stockfish_agents(white_stockfish_move, white_promoted)
            black_stockfish_move = self.black_agent.return_best_move()
            # Black to move
            black_valid_move, black_captured_piece, black_in_check, black_promoted = self.play(black_stockfish_move)
            self.update_stockfish_agents(black_stockfish_move, black_promoted) if black_valid_move else None
        else:
            print(f"white's move: {white_stockfish_move}, \n was it valid: {white_valid_move}\nis white in check? {white_in_check}, \nwhite king pos{self.white_agent.king_pos}, \nblack king pos: {self.black_agent.king_pos}, \n{self.white_agent.engine.get_board_visual()} \n{[row for row in self.board]}")
            black_valid_move, black_captured_piece, black_in_check, black_promoted = self.play(None)

        reward = self.calculate_reward(white_piece_moved, white_valid_move, white_captured_piece, white_in_check, black_valid_move, black_captured_piece, black_in_check)
        observation = self.state_space_generation()
        info = self.load_info()
        done = self.check_if_done() if (white_valid_move and black_valid_move) else True
        if done:
            print(f"mask: \n{observation['mask']}")
        if not self.is_training and self.wandb_run and done:
            self.render_episode()
        self.update_counter()
        return observation, reward, done, info

    def update_stockfish_agents(self, move, is_promoted):
        self.switch_player()
        try:
            self.white_agent.update_game_state(move, is_promoted)
            self.black_agent.update_game_state(move, is_promoted)
        except ValueError as e:
            print(f"{e} \n {move} \n {self.black_agent.engine.get_board_visual()}")

    def step_moveset(self, actions):
        _, _, _, _ = self.play(actions)
        self.switch_player()
        observation = self.state_space_generation()
        info = self.load_info()
        done = False
        self.update_counter()
        return observation, None, done, info

    def reset(self):
        self.board = initialize_board()
        self.virtual_states = []
        self.turn_counter = 1
        moveset_test = True if self._moveset_testing else False
        self.white_agent, self.black_agent = self.create_agents(color=None, board=self.board, stockfish_params=self.stockfish_params, moveset_test=moveset_test)
        self.current_player = "white"
        observation = self.state_space_generation()
        # Updating both stockfish agents
        if not self._moveset_testing:
            self.white_agent.update_skill_level(np.random.randint(20))
            self.black_agent.update_skill_level(np.random.randint(20))
        self.global_episode += 1
        return observation

    def set_moveset_test(self, mode):
        self._moveset_testing = mode
