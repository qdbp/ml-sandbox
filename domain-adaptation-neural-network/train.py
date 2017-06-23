import click
from keras.callbacks import ReduceLROnPlateau as LRC
from keras.callbacks import EarlyStopping, Callback
from keras.datasets import mnist
from keras.layers.convolutional import Convolution2D as C2D, MaxPooling2D as M2D
from keras.layers import Dense, Flatten, Input, Dropout
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.utils.np_utils import to_categorical
import numpy as np
from qqq.ml.keras_layers import NegGrad
from qqq.ml.keras_util import (ModelHandler, ProgLogger, WeightSaver,
                               apply_layers)

from qqq.ml.preprocess import gen_random_labels, get_k_of_each, complement_ixes


def get_main_stack():
    return [
        GaussianNoise(0.1),
        C2D(16, 3, 3, activation='relu', border_mode='valid'),
        C2D(16, 3, 3, activation='relu', border_mode='valid'),
        M2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', name='main_fc1'),
        Dropout(0.5),
        Dense(128, activation='relu', name='main_fc2'),
    ]


def mk_dann_model(num_false=10, dann_lambda=0.2):
    inp = Input(shape=(1, 28, 28))

    main_stack = get_main_stack()

    false_stacks = [
        [NegGrad(dann_lambda),
         # Dense(64, activation='relu', name=f'false{x}_fc0'),
         Dense(10, activation='softmax', name=f'false{x}_y'),
         ] for x in range(num_false)
    ]

    out_stack = [
        Dense(10, activation='softmax', name='true_y'),
    ]

    y_true = apply_layers(inp, main_stack, out_stack)
    false_ys = [apply_layers(inp, main_stack, ms) for ms in false_stacks]

    dann_model = Model(input=inp, output=[y_true] + false_ys)
    dann_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       loss_weights=[1] + [1/num_false] * num_false)

    return dann_model


def mk_plain_model():
    i = Input((1, 28, 28))

    main_stack = get_main_stack()

    out_stack = [
        Dense(10, activation='softmax'),
    ]

    y = apply_layers(i, main_stack, out_stack)
    plain_model = Model(input=i, output=y)
    plain_model.compile(loss='categorical_crossentropy',
                        optimizer='adam')

    return plain_model


@click.command()
@click.argument('cmd')
@click.option('--num-false', type=int, default=10)
@click.option('--lbd', type=float, default=0.2)
@click.option('--reset', default=False, is_flag=True)
@click.option('--reset-plain', default=False, is_flag=True)
@click.option('--only-plain', default=False, is_flag=True)
@click.option('--do-oneshot', default=False, is_flag=True)
@click.option('--oneshot-examples', default=10)
def main(cmd, *, num_false, lbd, reset, reset_plain, only_plain,
         do_oneshot, oneshot_examples):

    (X_train, y_train), (X_val, y_val) = mnist.load_data()

    y_train, y_val = to_categorical(y_train), to_categorical(y_val)

    X_train = (X_train/256 - 0.5)[:, np.newaxis, :]
    X_val = (X_val/256 - 0.5)[:, np.newaxis, :]

    if do_oneshot:
        train_ixes = get_k_of_each(y_train, oneshot_examples)
        val_ixes = complement_ixes(train_ixes, y_train)

        yt_os = y_train[train_ixes]
        yv_os = np.concatenate([y_val, y_train[val_ixes]])

        Xt_os = X_train[train_ixes]
        Xv_os = np.concatenate([X_val, X_train[val_ixes]])

        X_train, X_val = Xt_os, Xv_os
        y_train, y_val = yt_os, yv_os

    random_train = [gen_random_labels(X_train, 10) for i in range(num_false)]

    dann_model = mk_dann_model(num_false=num_false, dann_lambda=lbd)
    dann_model, handler = ModelHandler.attach(
        dann_model,
        name='one_shot_dann_mnist'
    )

    class LbdCallback(Callback):

        def on_epoch_end(self, epoch, logs):
            for layer in self.model.layers:
                if isinstance(layer, NegGrad):
                    layer.set_lbd(1 - 1/(1 + epoch/10))

    plain_model = mk_plain_model()
    plain_model, plain_handler = ModelHandler.attach(plain_model, name='plain')

    if cmd == 'train':
        if not only_plain:
            dann_model.fit(X_train, [y_train] + random_train,
                           nb_epoch=5000, batch_size=256,
                           callbacks=[LRC(patience=25),
                                      EarlyStopping(patience=100),
                                      ProgLogger(train_loss='true_y_loss',
                                                 val_loss='val_true_y_loss'),
                                      WeightSaver(load=not reset,
                                                  compare_key='val_true_y_loss'),
                                      LbdCallback()],
                           validation_split=0.2,
                           verbose=0)

        plain_model.fit(
            X_train, y_train, nb_epoch=5000, batch_size=256, verbose=0,
            callbacks=[
                LRC(patience=25), EarlyStopping(patience=100),
                ProgLogger(), WeightSaver(load=not reset_plain)],
            validation_split=0.2)


if __name__ == '__main__':
    main()
