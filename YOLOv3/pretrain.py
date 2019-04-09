from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, Concatenate, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.regularizers import l2
import scipy.io as scio
import numpy as np
import keras.backend as K
from yolo3.model import tiny_yolo_body, yolo_loss
from yolo3.utils import compose
from functools import wraps

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def tiny_yolo_body(input_shape=(128,1024,1), num_anchors=6, num_classes=4):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    inputs = Input(input_shape)
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])
    y1 = Flatten()(y1)
    y2 = Flatten()(y2)
    y1 = Dense(4, activation='softmax', name='y1', kernel_initializer=glorot_uniform(seed=0),
               kernel_regularizer=l2(0.0001))(y1)
    y2 = Dense(4, activation='softmax', name='y2', kernel_initializer=glorot_uniform(seed=0),
               kernel_regularizer=l2(0.0001))(y2)

    model = Model(inputs, [y1,y2], name='tinymodel')

    return model


model = tiny_yolo_body(input_shape=(128,1024,1), num_anchors=3, num_classes=4)

model.compile(optimizer='adam', loss={'y1': 'categorical_crossentropy', 'y2': 'categorical_crossentropy'}, metrics=['accuracy'])

Trainingset = scio.loadmat('pretraindata.mat')
X = Trainingset['Tx']
Y = Trainingset['Ty']

X_train = X[0:151]
Y_train = Y[0:151]
X_test = X[151:]
Y_test = Y[151:]

model.fit(X_train, {'y1': Y_train, 'y2': Y_train}, epochs=50, batch_size=16)
model.save_weights('pretraintiny.h5')
preds = model.evaluate(X_test, {'y1': Y_test, 'y2': Y_test})


