import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
import imageio
import seaborn as sns
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.layers import Dropout, BatchNormalization,LeakyReLU, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt




np.random.seed(11)
tensorflow.random.set_seed(11)





img_folder='./Dataset/bee_imgs/'
img_width=128
img_height=128
img_channels=3
bees=pd.read_csv('./Dataset/bee_data.csv', 
                index_col=False,  
                parse_dates={'datetime':[1,2]},
                dtype={'subspecies':'category', 'health':'category','caste':'category'})

def read_img(file):
    img = skimage.io.imread(img_folder + file)
    img = skimage.transform.resize(img, (img_width, img_height), mode='reflect')
    return img[:,:,:img_channels]

bees.dropna(inplace=True)
img_exists = bees['file'].apply(lambda f: os.path.exists(img_folder + f))
bees = bees[img_exists]




train_bees, test_bees = train_test_split(bees, random_state=65)
train_bees, val_bees = train_test_split(train_bees, test_size=0.1, random_state = 67)
ncat_bal = int(len(train_bees)/train_bees['health'].cat.categories.size)
train_bees_bal = train_bees.groupby('health', as_index=False).apply(lambda g:  g.sample(ncat_bal, replace=True)).reset_index(drop=True)
train_bees = train_bees_bal

print(train_bees)


print(test_bees)

train_X = np.stack(train_bees['file'].apply(read_img))
train_y  = pd.get_dummies(train_bees['health'], drop_first=False)

val_X = np.stack(val_bees['file'].apply(read_img))
val_y = pd.get_dummies(val_bees['health'], drop_first=False)

test_X = np.stack(test_bees['file'].apply(read_img))
test_y = pd.get_dummies(test_bees['health'], drop_first=False)



Data augmentation - a little bit rotate, zoom and shift input images.
generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)
generator.fit(train_X)







earlystopper = EarlyStopping(monitor='val_accuracy', patience=25, verbose=1)
checkpointer = ModelCheckpoint('./best_model.h5'
                                ,monitor='val_accuracy'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=True)
model=Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(img_width, img_height,3), activation='relu', padding='same'))
model.add(MaxPool2D(2))
model.add(Dropout(0.15))
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(train_y.columns.size, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps = np.round(train_X.shape[0] / 256, 0)
training = model.fit_generator(generator.flow(train_X,train_y, batch_size=256)
                        ,epochs=200
                        ,validation_data=(val_X, val_y)
                        ,steps_per_epoch=steps
                        ,callbacks=[earlystopper, checkpointer])






print('done training')
model.save('new__model.h5')
print('new__model created')

model.load_weights('./best_model.h5')
test_res = model.evaluate(test_X, test_y.values, verbose=0)
print('Loss function: %s, accuracy:' % test_res[0], test_res[1])
