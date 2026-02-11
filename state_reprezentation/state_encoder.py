import numpy as np

from deepdraughts.env.py_env.env_utils import RUSSIAN_RULE
from deepdraughts.env.py_env import Game

class StateEncoder:
    """
    Zoptymalizowana reprezentacja stanu:
    - Tensor (4, 4, 8) zamiast (10, 8, 8)
    - Tylko ciemne pola
    - Perspektywa gracza
    - Brak redundantnych kanałów
    """
    def __init__(self, board_size=8):
        self.board_size = board_size

    def game_to_tensor(self, game):
        """
        Zwraca tensor (6, 4, 8):

        Kanał 0: Moje pionki
        Kanał 1: Moje damki
        Kanał 2: Przeciwnika pionki
        Kanał 3: Przeciwnika damki
        Kanał 4: Możliwe ruchy (0=nie, 1=ruch, 2=bicie)
        Kanał 5: Możliwe cele ruchów (gdzie mogę pójść)
        """
        tensor = np.zeros((6, 4, self.board_size), dtype=np.float32)

        board = game.current_board
        my_player = game.current_player

        for pos, piece in board.pieces.items():
            if piece.captured:
                continue

            row, col, new_row, new_col = self._pos_to_coords(pos)

            is_mine = (piece.player == my_player)

            if is_mine:
                channel = 1 if piece.isking else 0
            else:
                channel = 3 if piece.isking else 2

            tensor[channel, new_row, new_col] = 1.0

        legal_moves = game.get_all_available_moves()
        for move in legal_moves:
            start_pos, _ = move.pos
            row, col, new_row, new_col = self._pos_to_coords(start_pos)

            is_capture = False
            if move.taken_pos is not None:
                is_capture = (isinstance(move.taken_pos, (list, tuple)) and len(move.taken_pos) > 0) or \
                             isinstance(move.taken_pos, int)

            tensor[4, new_row, new_col] = 2.0 if is_capture else 1.0

        for move in legal_moves:
            _, end_pos = move.pos
            row, col, new_row, new_col = self._pos_to_coords(end_pos)
            tensor[5, new_row, new_col] = 1.0

        return tensor

    def _pos_to_coords(self, pos):
        """
        Mapuje pozycję deepdraughts na skompresowaną siatkę 4×8

        Returns:
            row_orig, col_orig, row_new, col_new
        """
        row = pos // 8
        col = pos % 8

        new_row = row // 2

        if row % 2 == 0:
            new_col = col // 2
        else:
            new_col = col // 2 + 4
        return row, col, new_row, new_col

    def coords_to_pos(self, row, col):
        """
        Odwrotna konwersja: (row, col) -> pozycja

        Używana tylko dla ciemnych pól gdzie (row + col) % 2 == 1
        """
        if (row + col) % 2 == 0:
            raise ValueError(f"Pozycja ({row}, {col}) to jasne pole, nie ciemne!")
        return row * 8 + col

if __name__ == "__main__":
    game = Game(rule=RUSSIAN_RULE)
    encoder = StateEncoder()

    state = encoder.game_to_tensor(game)

    print(f"Kształt tensora: {state.shape}")
