import tensorflow
import skimage
import skimage.io
import skimage.transform
from keras.models import load_model
import numpy as np




img_folder='./Dataset/bee_imgs/'
img_width=128
img_height=128
img_channels=3

def read_img(file):
    img = skimage.io.imread(img_folder + file)
    img = skimage.transform.resize(img, (img_width, img_height), mode='reflect')
    return img[:,:,:img_channels]



image_location = read_img('032_633.png')

model = load_model('./new__model.h5')

# print(image_location)

print('\n')
print('\n')
print('\n')
print('\n')
print('\n')
print('\n')

image_location = np.array(image_location,ndmin=4)

# print(image_location)

prediction = model.predict_classes(image_location)

print(prediction)


