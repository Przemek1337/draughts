import numpy as np

from deepdraughts.env.py_env.env_utils import RUSSIAN_RULE
from deepdraughts.env.py_env import Game

class StateEncoder:
    """Konwerter stanu gry deepdraughts na tensor dla RL - wersja 8x8 (Russian)"""

    def __init__(self, board_size=8):
        self.board_size = board_size

    def game_to_tensor(self, game):
        """
        Konwertuje stan Game na tensor (C, H, W) dla planszy 8x8

        Args:
            game: obiekt Game z deepdraughts (reguły rosyjskie, 8x8)

        Returns:
            np.array: tensor (10, 8, 8) reprezentujący stan
        """
        tensor = np.zeros((10, self.board_size, self.board_size), dtype=np.float32)

        board = game.current_board

        for pos, piece in board.pieces.items():
            if piece.captured:
                continue

            row, col = self._pos_to_coords(pos)

            if piece.player == 1:
                if piece.isking:
                    tensor[1, row, col] = 1.0
                else:
                    tensor[0, row, col] = 1.0

            elif piece.player == -1:  # BLACK
                if piece.isking:
                    tensor[3, row, col] = 1.0
                else:
                    tensor[2, row, col] = 1.0

        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    tensor[4, row, col] = 1.0

        legal_moves = game.get_all_available_moves()
        for move in legal_moves:
            start_pos, end_pos = move.pos

            row, col = self._pos_to_coords(start_pos)
            tensor[5, row, col] = 1.0

            is_capture = False
            if move.taken_pos is not None:
                if isinstance(move.taken_pos, (list, tuple)):
                    is_capture = len(move.taken_pos) > 0
                else:
                    is_capture = True

            if is_capture:
                tensor[6, row, col] = 1.0

        current_player_value = 1.0 if game.current_player == 1 else 0.0
        tensor[7, :, :] = current_player_value

        white_pieces = np.sum(tensor[0:2, :, :])
        black_pieces = np.sum(tensor[2:4, :, :])
        tensor[8, :, :] = (white_pieces - black_pieces) / 24.0

        total_pieces = white_pieces + black_pieces
        tensor[9, :, :] = 1.0 - (total_pieces / 24.0)

        return tensor

    def _pos_to_coords(self, pos):
        """
        Konwertuje pozycję deepdraughts na współrzędne (row, col)

        deepdraughts używa numeracji: pos = row * 8 + col
        gdzie pos odpowiada tylko ciemnym polom (row + col) % 2 == 1
        """
        row = pos // 8
        col = pos % 8
        return row, col

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

    print("=" * 60)
    print("REPREZENTACJA STANU DLA REINFORCEMENT LEARNING")
    print("=" * 60)

    print(f"\nKonfiguracja:")
    print(f"  Reguły: Russian Draughts")
    print(f"  Rozmiar planszy: {game.current_board.ngrid} pól (8x8)")
    print(f"  Liczba pionków: {len(game.current_board.pieces)}")
    print(f"  Aktualny gracz: {'WHITE (1)' if game.current_player == 1 else 'BLACK (-1)'}")

    state_tensor = encoder.game_to_tensor(game)

    print(f"\n" + "=" * 60)
    print("TENSOR STANU:")
    print("=" * 60)
    print(f"Kształt: {state_tensor.shape} (kanały, wysokość, szerokość)")
