import os
import numpy as np
from othello.OthelloUtil import getValidMoves
from othello.bots.DeepLearning.OthelloModel import OthelloModel
from othello.OthelloGame import OthelloGame
from keras.callbacks import EarlyStopping, ModelCheckpoint

class BOT():
    def __init__(self, board_size, *args, **kargs):
        self.board_size = board_size
        self.collect_gaming_data = False
        self.history = []
        self.epsilon = 0.1  # 初始隨機性參數
        self.epsilon_min = 0.01  # 最小隨機性參數
        self.epsilon_decay = 0.995  # 隨機性衰減參數
        self.model = OthelloModel(input_shape=(self.board_size, self.board_size))
        # 嘗試加載模型權重，如果不存在則重新訓練
        try:
            self.model.load_weights()
            print('model loaded')
        except:
            print('no model exist')

    def getAction(self, game, color):
        # 使用神經網絡模型選擇行動
        predict = self.model.predict(game)
        valid_positions = getValidMoves(game, color)
        valids = np.zeros((game.size), dtype='int')
        valids[[i[0] * self.board_size + i[1] for i in valid_positions]] = 1
        predict *= valids

        # 引入隨機性
        if np.random.rand() < self.epsilon:
            position = np.random.choice(np.flatnonzero(valids))
        else:
            position = np.argmax(predict)

        if self.collect_gaming_data:
            tmp = np.zeros_like(predict)
            tmp[position] = 1.0
            self.history.append([np.array(game.copy()), tmp, color])

        position = (position // self.board_size, position % self.board_size)

        # 每次選擇行動後減少 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return position

    def self_play_train(self, args):
        iteration = 0  # 初始化 iteration 變數

        while True:
            iteration += 1
            if args['verbose']:
                print(f'self playing iteration {iteration}')

            self.collect_gaming_data = True  # 確保每次迭代開始時收集數據

            def gen_data():
                def getSymmetries(board, pi):
                    pi_board = np.reshape(pi, (len(board), len(board)))
                    l = []
                    for i in range(1, 5):
                        for j in [True, False]:
                            newB = np.rot90(board, i)
                            newPi = np.rot90(pi_board, i)
                            if j:
                                newB = np.fliplr(newB)
                                newPi = np.fliplr(newPi)
                            l += [(newB, list(newPi.ravel()))]
                    return l

                self.history = []
                history = []
                game = OthelloGame(self.board_size)
                game.play(self, self, verbose=args['verbose'])
                
                for step, (board, probs, player) in enumerate(self.history):
                    sym = getSymmetries(board, probs)
                    for b, p in sym:
                        history.append([b, p, player])
                self.history.clear()
                
                game_result = game.isEndGame()

                # 過濾無效數據
                return [(x[0], x[1]) for x in history if (game_result == 0 or x[2] == game_result)]

            data = []
            
            for i in range(args['num_of_generate_data_for_train']):
                if args['verbose']:
                    print('self playing', i+1)
                data += gen_data()
            
            self.collect_gaming_data = False  # 訓練開始前停止收集數據

            if args['verbose']:
                print(f"Training on {len(data)} samples")

            # 添加 ModelCheckpoint 回調函數
            checkpoint = ModelCheckpoint('othello/bots/DeepLearning/models/model_10x10.h5', monitor='loss', save_best_only=True, mode='min', verbose=1)
            callbacks = [checkpoint]

            self.model.fit(data, batch_size=args['batch_size'], epochs=args['epochs'], callbacks=callbacks)

            # 減少 epsilon 確保模型逐步專注於最佳策略
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
