from utils.stockfish_utils import convert_to_action_space, convert_to_stockfish_space
from agents.chess_agent import ChessAgent

from stockfish import Stockfish


class StockfishAgent(ChessAgent):
    def __init__(self, color, params):
        super().__init__(color)
        self.params = params
        self.wait_time = params["wait_time"]
        self.engine = Stockfish(path=params["binary"], depth=params["depth"], parameters=params["engine_params"])
    
    def return_best_move(self):
        best_move = self.engine.get_best_move_time(self.wait_time)
        return best_move if best_move == None else convert_to_action_space(best_move)
    
    def update_game_state(self, move, is_promoted):
        self.engine.make_moves_from_current_position(convert_to_stockfish_space(move, is_promoted))

    def update_skill_level(self, level):
        self.engine.set_skill_level(level)
    
    def get_evaluation(self):
        evaluation = self.engine.get_evaluation()
        print(evaluation)

    def close(self):
        self.engine.send_quit_command()
        