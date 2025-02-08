import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.layers import AvgPool2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.constraints import maxnorm


LOSS = "categorical_crossentropy"  # Loss function
NUMOFCLASSES = 10  # Number of classes
lr = 0.0025
OPTIMIZER = SGD(
    lr=lr, momentum=0.9, decay=lr / (EPOCHS * CLIENT_EPOCHS), nesterov=False
)  # lr = 0.015, 67 ~ 69%


class Model:

    def __init__(self, loss, optimizer, classes=10):
        self.loss = loss
        self.optimizer = optimizer
        self.num_classes = classes

    def fl_paper_model(self, train_shape):
        model = Sequential()

        # 1
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding="same",
                activation="relu",
                input_shape=train_shape,
                kernel_regularizer="l2",
            )
        )
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding="same",
                activation="relu",
                kernel_regularizer="l2",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(0.2))

        # 2
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(5, 5),
                padding="same",
                activation="relu",
                kernel_regularizer="l2",
            )
        )
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(5, 5),
                padding="same",
                activation="relu",
                kernel_regularizer="l2",
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(0.2))

        # 3
        model.add(Flatten())
        model.add(
            Dense(
                units=512,
                activation="relu",
                kernel_regularizer="l2",
            )
        )
        model.add(Dropout(0.2))

        # 4
        model.add(Dense(units=self.num_classes, activation="softmax"))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])

        return model


model = Model(loss=LOSS, optimizer=OPTIMIZER, classes=NUMOFCLASSES)
init_model = model.fl_paper_model(train_shape=10)

print(init_model.get_weights())
