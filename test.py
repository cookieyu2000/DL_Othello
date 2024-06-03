from othello.OthelloGame import OthelloGame
from othello.bots.Random import BOT as RandomBOT
from othello.bots.DeepLearning.mcts_treenode import MCTS_BOT



BLACK = 1
WHITE = -1

BOARD_SIZE = 10

def self_play(black, white, verbose=True):
    g = OthelloGame(n=BOARD_SIZE)
    result = g.play(black, white, verbose)
    return result


def main():
    n_game = 10
    bot1_win = 0
    bot2_win = 0

    bot1_name = "RandomBOT"
    bot1 = RandomBOT()

    bot2_name = "MCTS_BOT"
    bot2 = MCTS_BOT(c_uct=3, n_playout=300, n=BOARD_SIZE, time_limit=2, name=bot2_name)

    for i in range(n_game):
        print("Game {}".format(i+1))
        result = self_play(bot1, bot2, verbose=False)
        if result == BLACK:
            bot1_win += 1
        elif result == WHITE:
            bot2_win += 1

        result = self_play(bot2, bot1, verbose=True)
        if result == BLACK:
            bot2_win += 1
        elif result == WHITE:
            bot1_win += 1
        with open("result.txt", "a") as f:
            f.write("Game {}\n".format(i+1))
            f.write("{} win: {}\n".format(bot1_name, bot1_win))
            f.write("{} win: {}\n".format(bot2_name, bot2_win))
            f.write(
                "----------------------------------------------------------------------------\n")
        print("{} win: {}".format(bot1_name, bot1_win))
        print("{} win: {}".format(bot2_name, bot2_win))
        print(
            "----------------------------------------------------------------------------")


if __name__ == '__main__':

    main()
