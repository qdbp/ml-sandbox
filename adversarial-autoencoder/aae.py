'''
Playing around with an adversarial autoencoder, as in 

Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint
arXiv:1511.05644 (2015).
'''

import os

import keras.callbacks as kcb
import keras.layers.advanced_activations as kra
import keras.layers as klc
import keras.layers.noise as kln
import keras.layers.normalization as klm
import keras.regularizers as krr
import keras.models as krm
import keras.optimizers as kro
from keras.datasets import mnist

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as pch
import numpy as np
import numpy.random as npr

import qqq.util as qqu
import qqq.io as qio


batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = np.vstack([X_train, X_test])
s = np.std(X_train)
m = np.mean(X_train)
X_train = (X_train - m)/s

dirname = 'aae'
os.makedirs(dirname, exist_ok=True)

H_SIZE_GAUS = 6
H_SIZE_CAT = 0
H_SIZE = H_SIZE_GAUS + H_SIZE_CAT
BS = 256
NOISE_SCALE = 5

# separate encoder learning rate for pinning
ENC_LR_BASE = 0
AE_LR_BASE = 1e-4
H_DEC_LR_BASE = 1e-5

i_enc = klc.Input(shape=(784,), name='enc_input')
i_gen = klc.Input(shape=(H_SIZE_GAUS,), name='gen_input')
i_dec_h = klc.Input(shape=(H_SIZE_GAUS,), name='dec_h_input')
i_dec = klc.Input(shape=(784,), name='dec_input')


# ENCODER
enc_ls =\
    [klc.GaussianNoise(0.2),
     klc.Dense(1024, name='enc_d0', activation='relu'),
     klc.Dropout(0.25, name='enc_do0'),
     klc.Dense(1024, name='enc_d1', activation='relu'),
     klc.Dropout(0.25, name='enc_do1'),
     klc.Dense(H_SIZE, name='enc_d3_y', activation='linear')]

# GENERATOR
gen_ls =\
    [klc.Dense(1024, name='gen_d0', activation='relu'),
     klc.Dropout(0.25, name='gen_do0'),
     klc.Dense(1024, name='gen_d1', activation='relu'),
     klc.Dropout(0.25, name='gen_do1'),
     klc.Dense(784, name='gen_d3_y', activation='linear')]

# DECODER_HIDDEN
dec_h_ls =\
    [kln.GaussianNoise(0.1),
     klc.Dropout(0.25, name='dec_h_do0'),
     klc.Dense(100, name='dec_h_d1', activation='relu'),
     klc.Dropout(0.25, name='dec_h_do2'),
     klc.Dense(100, name='dec_h_d2', activation='relu'),
     klc.Dense(2, name='dec_d3_h_y', activation='softmax')]

# MODELS

# AE
#

ae_out = i_enc
for l in enc_ls + gen_ls:
    ae_out = l(ae_out)

m_ae = krm.Model(input=i_enc, output=ae_out)
ae_opti = kro.Adam(lr=AE_LR_BASE)
m_ae.compile(optimizer=ae_opti, loss='mse')

# lower GAN
#
enc_out = i_enc
for l in enc_ls:
    enc_out = l(enc_out)

full_h_out = i_enc
for l in enc_ls + dec_h_ls:
    full_h_out = l(full_h_out)

dec_h_out = i_dec_h
for l in dec_h_ls:
    dec_h_out = l(dec_h_out)

m_enc = krm.Model(input=i_enc, output=enc_out)
enc_opti = kro.Adam(lr=ENC_LR_BASE)
m_enc.compile(optimizer=enc_opti, loss='mse')

m_full_h = krm.Model(input=i_enc, output=full_h_out)
full_h_opti = kro.Adam(lr=1e-5)
m_full_h.compile(optimizer=full_h_opti, loss='categorical_crossentropy')

m_dec_h = krm.Model(input=i_dec_h, output=dec_h_out)
dec_h_opti = kro.Adam(lr=H_DEC_LR_BASE)
m_dec_h.compile(optimizer=dec_h_opti, loss='categorical_crossentropy')

# training specific 
true_ys = np.zeros((BS, 2))
false_ys = np.zeros((BS, 2))

h_dec_loss_bias = 0
i_h_dec_loss_bias = 0

k = 0.01

m_ae.fit(X_train, X_train, nb_epoch=5, batch_size=256)

# 1-point-pin experiment
null_pin = X_train[npr.randint(len(X_train))][np.newaxis, :]
one_pin = X_train[npr.randint(len(X_train))][np.newaxis, :]

test_ixes = npr.randint(len(X_train), size=1024)
test_enc_input = X_train[test_ixes]
test_enc_input[0] = null_pin[0]
test_enc_input[1] = one_pin[0]

one_pin_true = np.asarray([[5, 0, 0, 0, 0, 0]])

import matplotlib.animation as man
FFMpegWriter = man.writers['ffmpeg']
metadata = dict(title='Autism Production Presents: Walts of the zs',
                artist='xX_d4nk_k1ll3r_n4um0v_CLXX_xXx')
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure()
ax = fig.add_subplot(111)

with writer.saving(fig, dirname+"/double_pinned_6D_xxxtralow_lr_waltz_of_the_zs.mp4", 100):
    for e in range(10000):
        true_ys[:, 0] = 1
        false_ys[:, 1] = 1
        
        ## autoencoder training pass
        ixes_ae = npr.randint(len(X_train), size=BS)
        real_ae = X_train[ixes_ae]
        ae_loss = m_ae.train_on_batch(real_ae, real_ae)

        # pinning pass
        zero_pin_loss = m_enc.train_on_batch(null_pin, np.zeros((1, H_SIZE)))
        one_pin_loss = m_enc.train_on_batch(null_pin, one_pin_true)
    
        ## encoder/hidden decoder training pass
        # encoder training pass
        for l in dec_h_ls:
            l.trainable = False
    
        ixes_enc = npr.randint(len(X_train), size=BS)
        real_enc = X_train[ixes_enc]
    
        encoder_loss = m_full_h.train_on_batch(real_enc, true_ys)
    
        # hidden decoder training pass
        for l in dec_h_ls:
            l.trainable = True
    
        ixes_enc = npr.randint(len(X_train), size=BS)
        enc_input = X_train[ixes_enc]
    
        fake_enc = m_enc.predict(enc_input)
        real_enc = npr.normal(scale=NOISE_SCALE, size=(BS, H_SIZE_GAUS))
    
        mix_X_enc = np.vstack([real_enc, fake_enc])
        mix_y_enc = np.vstack([true_ys, false_ys])
        
        rs = npr.get_state()
        npr.shuffle(mix_X_enc)
        npr.set_state(rs)
        npr.shuffle(mix_y_enc)
    
        h_decoder_loss = m_dec_h.train_on_batch(mix_X_enc, mix_y_enc)
    
        h_dec_loss_bias = k*(h_decoder_loss - encoder_loss) + (1-k)*h_dec_loss_bias
        i_h_dec_loss_bias += 3e-3*h_dec_loss_bias
    
        if not (e % 10):
            dec_h_opti.lr.set_value(
                    np.asarray(H_DEC_LR_BASE*np.exp(h_dec_loss_bias +
                                                    i_h_dec_loss_bias),
                               dtype=np.float32)
                    )
    
        qio.wr('E{}: '
               'ae_loss {:7.5f} [ae ulr {:5.4f}] | '
               'h_dec_loss {:7.5f}, enc_loss {:7.5f} [h_dec ulr {:7.4f}] | '
               'h_dec_loss_bias[i] {:7.5f}[{:7.5f}] |'
               'zpl, opl: {:4.3f}, {:4.3f}'
               .format(e,
                       float(ae_loss), 1e6*float(ae_opti.lr.get_value()),
                       float(h_decoder_loss), float(encoder_loss),
                       1e6*float(dec_h_opti.lr.get_value()),
                       h_dec_loss_bias, i_h_dec_loss_bias,
                       float(zero_pin_loss), float(one_pin_loss)
                       )
               )
    
        if not (e%5):
            pred = m_enc.predict(test_enc_input)
    
            ax.scatter(pred[1:, 0], pred[1:, 1])
            ax.scatter(pred[:1, 0], pred[:1, 1], color='k')
            # ax.scatter(npr.normal(scale=NOISE_SCALE, size=256),
            #            npr.normal(scale=NOISE_SCALE, size=256),
            #            color='red')
            circle1 = pch.Circle((0, 0), radius=5, transform=ax.transData,
                    fill=False)
            circle2 = pch.Circle((0, 0), radius=10, transform=ax.transData,
                    fill=False)
            circle3 = pch.Circle((0, 0), radius=15, transform=ax.transData,
                    fill=False)

            ax.add_artist(circle1)
            ax.add_artist(circle2)
            ax.add_artist(circle3)

            ax.set_xlim((-15, 15))
            ax.set_ylim((-15, 15))
            # plt.savefig(dirname+'/{:06d}.png'.format(e))
            writer.grab_frame()
            plt.cla()
            
