import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import Model
import os

import params

MAIN_DIR = os.getcwd()
INPUT_DIR = os.path.join(MAIN_DIR,'input')
TRAIN_DIR = os.path.join(INPUT_DIR,'train')
MASK_DIR = os.path.join(INPUT_DIR,'mask')

input_size = params.input_size
epochs = params.max_epochs
batch_size = params.batch_size
model = params.model_factory()

os.chdir(INPUT_DIR)
df_train = pd.read_csv('train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                path = os.path.join(TRAIN_DIR,'{}.png'.format(id))
                img = cv2.imread(path)
                path = os.path.join(MASK_DIR,'{}.png'.format(id))
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x = np.array(x_batch, np.float32)
            x = x/255
            y = np.array(y_batch, np.float32)
            y=y/255
            yield x, y


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                path = os.path.join(TRAIN_DIR,'{}.png'.format(id))
                img = cv2.imread(path)
                path = os.path.join(MASK_DIR,'{}.png'.format(id))
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x = np.array(x_batch, np.float32)
            x = x/255
            y = np.array(y_batch, np.float32)
            y=y/255
            yield x, y

os.chdir(MAIN_DIR)

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='best_weights_256.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))

