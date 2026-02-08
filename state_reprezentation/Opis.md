Nasz tensor ma kształt (10, 8, 8):

10 = liczba kanałów (warstw informacji)
8×8 = rozmiar planszy

Każdy kanał to osobna "warstwa" informacji:
```aiignore
Kanał 0: Gdzie są BIAŁE PIONKI (1.0 = jest pionek, 0.0 = nie ma)
Kanał 1: Gdzie są BIAŁE DAMKI
Kanał 2: Gdzie są CZARNE PIONKI
Kanał 3: Gdzie są CZARNE DAMKI
Kanał 4: Które pola są DOSTĘPNE (ciemne pola, na których można grać)
Kanał 5: Z których pól można WYKONAĆ RUCH
Kanał 6: Z których pól można wykonać BICIE
Kanał 7: CZYJ RUCH (1.0 = białe, 0.0 = czarne)
Kanał 8: BILANS MATERIAŁU (różnica liczby pionków)
Kanał 9: FAZA GRY (jak daleko jesteśmy w grze)
```

Uzycie:

```aiignore
# 1. Inicjalizacja
from deepdraughts.env.py_env.env_utils import RUSSIAN_RULE
from deepdraughts.env.py_env import Game
from state_reprezentation.state_encoder import StateEncoder

game = Game(rule=RUSSIAN_RULE)
encoder = StateEncoder()

# 2. Konwersja stanu na tensor (dla NN)
state = encoder.game_to_tensor(game)  # Shape: (10, 8, 8)

# 3. W pętli treningowej
while not game.game_is_over():
    # Pobierz stan
    state = encoder.game_to_tensor(game)
    
    #sieć neuronowa przewiduje akcję
    action = your_model.predict(state)
    
    # Wykonaj ruch
    moves = game.get_all_available_moves()
    game.do_move(moves[action])
    
    # Nowy stan
    next_state = encoder.game_to_tensor(game)
```