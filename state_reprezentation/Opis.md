Zwraca tensor (6, 4, 8):

10 = liczba kanałów (warstw informacji)
8×8 = rozmiar planszy

Każdy kanał to osobna "warstwa" informacji:
```aiignore
Kanał 0: Moje pionki
Kanał 1: Moje damki
Kanał 2: Przeciwnika pionki
Kanał 3: Przeciwnika damki
Kanał 4: Możliwe ruchy (0=nie, 1=ruch, 2=bicie)
Kanał 5: Możliwe cele ruchów (gdzie mogę pójść)
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