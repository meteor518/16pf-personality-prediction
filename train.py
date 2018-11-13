# coding:utf-8

import keras
import pandas as pd
import argparse
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Activation, GlobalMaxPooling2D, Dropout, \
    GlobalAveragePooling2D, BatchNormalization
from keras.layers import Conv2D, LocallyConnected2D, Concatenate, concatenate, Input, Merge
from keras.callbacks import *
from keras.optimizers import rmsprop, Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from skimage import color, exposure, transform
from PIL import Image
from metrics import precision, recall, fmeasure
from vgg_face import vgg_face
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ['KERAS_BACKEND'] = "tensorflow"

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 进行配置，使用70%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)


def label_encode(nlabels, label_value):
    label_value = keras.utils.to_categorical(label_value, nlabels)
    return label_value


def label_transform(df_label):
    y_train_Warmth = [i for i in df_label["gregarious"]]
    y_train_Reasoning = [i for i in df_label["intelligent"]]
    y_train_Emotional_Stability = [i for i in df_label["stable"]]
    y_train_Dominance = [i for i in df_label["dominate"]]
    y_train_Liveliness = [i for i in df_label["lively"]]
    y_train_Rule_Consciousness = [i for i in df_label["justic"]]
    y_train_Social_Boldness = [i for i in df_label["courage"]]
    y_train_Sensitivity = [i for i in df_label["sensitive"]]
    y_train_Vigilance = [i for i in df_label["suspicious"]]
    y_train_Abstractedness = [i for i in df_label["pragmatic"]]
    y_train_Privateness = [i for i in df_label["sophisticated"]]
    y_train_Apprehension = [i for i in df_label["safely"]]
    y_train_Openness_to_Change = [i for i in df_label["change"]]
    y_train_Self_Reliance = [i for i in df_label["independent"]]
    y_train_Perfectionism = [i for i in df_label["selfcontrol"]]
    y_train_Tension = [i for i in df_label["gentle"]]

    return np.array(y_train_Warmth), np.array(y_train_Reasoning), np.array(y_train_Emotional_Stability), np.array(
        y_train_Dominance), np.array(y_train_Liveliness), np.array(y_train_Rule_Consciousness), np.array(
        y_train_Social_Boldness), np.array(y_train_Sensitivity), np.array(y_train_Vigilance), np.array(
        y_train_Abstractedness), np.array(y_train_Privateness), np.array(y_train_Apprehension), np.array(
        y_train_Openness_to_Change), np.array(y_train_Self_Reliance), np.array(y_train_Perfectionism), np.array(
        y_train_Tension)

def custom_log_func(tensorboard, epoch, logs=None):
    opt = tensorboard.model.optimizer
    opt_lr = K.eval(opt.lr)
    return {"learning_rate": opt_lr}


def my_generator(x_train, df_label, batch_size):
    datagen = ImageDataGenerator(
        # rotation_range = 30,
        # width_shift_range = 0.2,
        # height_shift_range = 0.2,
        rescale=1.0 / 255.
        # horizontal_flip = True
    )
    datagen.fit(x_train)

    y_train_Warmth = [i for i in df_label["gregarious"]]
    y_train_Reasoning = [i for i in df_label["intelligent"]]
    y_train_Emotional_Stability = [i for i in df_label["stable"]]
    y_train_Dominance = [i for i in df_label["dominate"]]
    y_train_Liveliness = [i for i in df_label["lively"]]
    y_train_Rule_Consciousness = [i for i in df_label["justic"]]
    y_train_Social_Boldness = [i for i in df_label["courage"]]
    y_train_Sensitivity = [i for i in df_label["sensitive"]]
    y_train_Vigilance = [i for i in df_label["suspicious"]]
    y_train_Abstractedness = [i for i in df_label["pragmatic"]]
    y_train_Privateness = [i for i in df_label["sophisticated"]]
    y_train_Apprehension = [i for i in df_label["safely"]]
    y_train_Openness_to_Change = [i for i in df_label["change"]]
    y_train_Self_Reliance = [i for i in df_label["independent"]]
    y_train_Perfectionism = [i for i in df_label["selfcontrol"]]
    y_train_Tension = [i for i in df_label["gentle"]]

    genx1 = datagen.flow(x_train, y_train_Warmth, batch_size=batch_size, seed=42)
    genx2 = datagen.flow(x_train, y_train_Reasoning, batch_size=batch_size, seed=42)
    genx3 = datagen.flow(x_train, y_train_Emotional_Stability, batch_size=batch_size, seed=42)
    genx4 = datagen.flow(x_train, y_train_Dominance, batch_size=batch_size, seed=42)
    genx5 = datagen.flow(x_train, y_train_Liveliness, batch_size=batch_size, seed=42)
    genx6 = datagen.flow(x_train, y_train_Rule_Consciousness, batch_size=batch_size, seed=42)
    genx7 = datagen.flow(x_train, y_train_Social_Boldness, batch_size=batch_size, seed=42)
    genx8 = datagen.flow(x_train, y_train_Sensitivity, batch_size=batch_size, seed=42)
    genx9 = datagen.flow(x_train, y_train_Vigilance, batch_size=batch_size, seed=42)
    genx10 = datagen.flow(x_train, y_train_Abstractedness, batch_size=batch_size, seed=42)
    genx11 = datagen.flow(x_train, y_train_Privateness, batch_size=batch_size, seed=42)
    genx12 = datagen.flow(x_train, y_train_Apprehension, batch_size=batch_size, seed=42)
    genx13 = datagen.flow(x_train, y_train_Openness_to_Change, batch_size=batch_size, seed=42)
    genx14 = datagen.flow(x_train, y_train_Self_Reliance, batch_size=batch_size, seed=42)
    genx15 = datagen.flow(x_train, y_train_Perfectionism, batch_size=batch_size, seed=42)
    genx16 = datagen.flow(x_train, y_train_Tension, batch_size=batch_size, seed=42)

    while True:
        xi1 = genx1.next()
        xi2 = genx2.next()
        xi3 = genx3.next()
        xi4 = genx4.next()
        xi5 = genx5.next()
        xi6 = genx6.next()
        xi7 = genx7.next()
        xi8 = genx8.next()
        xi9 = genx9.next()
        xi10 = genx10.next()
        xi11 = genx11.next()
        xi12 = genx12.next()
        xi13 = genx13.next()
        xi14 = genx14.next()
        xi15 = genx15.next()
        xi16 = genx16.next()
        yield xi1[0], [xi1[1], xi2[1], xi3[1], xi4[1], xi5[1], xi6[1], xi7[1], xi8[1], xi9[1], xi10[1], xi11[1],
                       xi12[1], xi13[1], xi14[1], xi15[1], xi16[1]]


def main(args):
    
    prefix = ['gregarious', 'intelligent', 'stable', 'dominate', 'lively', 'justice', 'courage', 'sensitive',
          'suspicious', 'pragmatic', 'sophisticated', 'safely', 'change', 'independent', 'selfcontrol', 'gentle']
    
    # images
    x_train = np.load(args.train)    
    x_train = x_train / 255.0    
    x_train = np.transpose(x_train, (0, 3, 1, 2))    
    
    # label
    y_train = pd.read_csv(args.train_label)    
    y_train = np.array(args.train_label[prefix])    
    y_train = list(y_train.T)
    print('train images shape:', x_train.shape, 'train labels shape', np.shape(y_train))
    
    if args.val:
         x_val = np.load(args.val)
         x_val = x_val / 255.0
         x_val = np.transpose(x_val, (0, 3, 1, 2))
         
         y_val = pd.read_csv(args.val_label)
         y_val = np.array(args.val_label[prefix])
         y_val = list(y_val.T)
         print('val images shape:', x_val.shape, 'val labels shape', np.shape(y_val))
        
    # model
    model = vgg_face()
    fc5 = model.layers[-8].output
    fc6 = Flatten()(fc5)
    fc7_1 = Dense(256, activation='relu', name='fc7_1')(fc6)
    dropout7_1 = Dropout(0.3)(fc7_1)
    fc7_2 = Dense(128, activation='relu', name='fc7_2')(dropout7_1)
    prediction = []
    for i in range(16):
        prediction.append(Dense(10, activation="softmax", name=prefix[i])(fc7_2))
    model = Model(inputs=model.input, outputs=prediction)

    print(model.summary())
    
    # model callbacks
    checkpoint_prefix = "vggface"
    os.makedirs(args.output, exist_ok=True)
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy", precision, recall, fmeasure])
    checkpoint = ModelCheckpoint(filepath=args.output + checkpoint_prefix + "-{epoch:02d}-{loss:.5f}-{val_loss:.5f}")
    tf_board = TensorBoard(log_dir=args.output + "tf_logs/", histogram_freq=0, write_graph=True)
    csv = CSVLogger(args.output+ 'log/' + checkpoint_prefix + '.csv')
    
    if args.val:
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=args.batch_size,
                  callbacks=[checkpoint, tf_board, csv], epochs=args.epochs, verbose=2)
    else:
        model.fit(x_train, y_train, batch_size=args.batch_size, callbacks=[checkpoint, tf_board, csv], 
                  epochs=args.epochs, verbose=2)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--train', '-t', required=True, help='the .npy file of train images')
    parse.add_argument('--val', '-v', help='the .npy file of validation images')
    parse.add_argument('--train-label', '-tl', required=True, help='the .csv file of train labels')
    parse.add_argument('--val-label', '-vl', help='the .csv file of validation labels')
    parse.add_argument('--output', '-o', default='./', help='the output path')
    
    parse.add_argument('--batch-size', '-batch', type=int, default=32)
    parse.add_argument('--epochs', '-e', type=int, default=100)
    args = parse.parse_args()
    
    main(args)
