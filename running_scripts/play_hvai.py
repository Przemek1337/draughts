from deepdraughts.env import HUMAN_PLAYER, AI_PLAYER
from deepdraughts.gui import GUI
from deepdraughts.mcts_pure import MCTSPlayer

def main():
    """
    Ty vs mcts(są inne modele)
    """
    gui = GUI()
    ai_black = MCTSPlayer(n_playout=1000)

    gui.run(
        player_white=HUMAN_PLAYER,
        player_black=AI_PLAYER,
        policy_white=None,
        policy_black=ai_black,
    )

if __name__ == "__main__":
    main()
