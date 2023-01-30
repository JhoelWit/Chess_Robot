import numpy as np


def convert_to_action_space(move):
    from_col = ord(move[0]) - ord('a')
    from_row = 8 - int(move[1])
    to_col = ord(move[2]) - ord('a')
    to_row = 8 - int(move[3])
    # print(f"converted {move} to {from_row, from_col, to_row, to_col}")
    return np.array([from_row, from_col, to_row, to_col])

def convert_to_stockfish_space(action, is_promoted):
    from_col = chr(action[1] + ord('a'))
    from_row = str(8 - action[0])
    to_col = chr(action[3] + ord('a'))
    to_row = str(8 - action[2])
    promoted_piece = "Q" if is_promoted else ""
    # print(f"converted {action} to {from_col + from_row + to_col + to_row}")
    return [from_col + from_row + to_col + to_row + promoted_piece]
