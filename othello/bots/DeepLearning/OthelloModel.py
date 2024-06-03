# othello/bots/DeepLearning/OthelloModel.py
import numpy as np, os
from keras.models import Model 
from keras.layers import ( 
    Input, Reshape, GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization,
    Activation, add, Dropout
)
from keras.optimizers import Adam 
from keras.regularizers import l2 
from keras.callbacks import EarlyStopping 
from tensorflow import summary


class OthelloModel():
    def __init__(self, input_shape=(10, 10)):
        self.model_name='model_'+( 'x'.join(list(map(str, input_shape))) )+'.h5'
        self.input_boards = Input(shape=input_shape)
        x_image = Reshape(input_shape + (1,))(self.input_boards)
        resnet_v12 = self.resnet_v1(inputs=x_image, num_res_blocks=10)
        gap1 = GlobalAveragePooling2D()(resnet_v12)
        gap1 = Dropout(0.5)(gap1)
        self.pi = Dense(input_shape[0] * input_shape[1], activation='softmax', name='pi',
                        kernel_regularizer=l2(1e-4))(gap1)
        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(0.001))
        # self.model.summary()

    def predict(self, board):
        return self.model.predict(np.array([board]).astype('float32'))[0]

    def fit(self, data, batch_size, epochs, callbacks=[]):
        if len(data) == 0:
            print(f"Data: {data}")
            raise ValueError("The data provided to fit is empty")
        
        input_boards, target_policys = list(zip(*data))
        input_boards = np.array(input_boards)
        target_policys = np.array(target_policys)
        self.model.fit(x=input_boards, y=[target_policys], batch_size=batch_size, epochs=epochs, callbacks=callbacks)



    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_weights(self):
        self.model.save_weights('othello/bots/DeepLearning/models/'+self.model_name)

    def load_weights(self):
        self.model.load_weights('othello/bots/DeepLearning/models/'+self.model_name)

    def reset(self, confirm=False):
        if not confirm:
            raise Exception('this operate would clear model weight, pass confirm=True if really sure')
        else:
            try:
                os.remove('othello/bots/DeepLearning/models/' + self.model_name)
            except:
                pass
        print('cleared')

    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for i in range(num_res_blocks):
            resnet = self.resnet_layer(inputs=x, num_filter=128)
            resnet = self.resnet_layer(inputs=resnet, num_filter=128, activation=None)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(num_res_blocks):
            if i == 0:
                resnet = self.resnet_layer(inputs=x, num_filter=256, strides=2)
                resnet = self.resnet_layer(inputs=resnet, num_filter=256, activation=None)
            else:
                resnet = self.resnet_layer(inputs=x, num_filter=256)
                resnet = self.resnet_layer(inputs=resnet, num_filter=256, activation=None)
            if i == 0:
                x = self.resnet_layer(inputs=x, num_filter=256, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(num_res_blocks):
            if i == 0:
                resnet = self.resnet_layer(inputs=x, num_filter=512, strides=2)
                resnet = self.resnet_layer(inputs=resnet, num_filter=512, activation=None)
            else:
                resnet = self.resnet_layer(inputs=x, num_filter=512)
                resnet = self.resnet_layer(inputs=resnet, num_filter=512, activation=None)
            if i == 0:
                x = self.resnet_layer(inputs=x, num_filter=512, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet
        return x

    def resnet_layer(self, inputs, num_filter=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True, padding='same'):
        conv = Conv2D(num_filter, kernel_size=kernel_size, strides=strides, padding=padding,
                      use_bias=False, kernel_regularizer=l2(1e-4))
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x
    

