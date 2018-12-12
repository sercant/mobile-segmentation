import keras
from keras import backend as K
from keras import layers

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from nets.deeplabv3p import network
from datasets.coco_dataset import DataGenerator as coco_generator

import imgaug
augmentation = imgaug.augmenters.Fliplr(0.5)


K.set_learning_phase(1)
input_tensor = layers.Input(
    shape=(224, 224, 3), name='input_1')

# net = MobileUNet_v2(
#     input_tensor=input_tensor,
#     classes=1
# )
model = network(input_tensor, num_classes=3)

# Load weights.
model.load_weights('./dist/weights_deep.h5')
# equivalent but more general
generator = coco_generator(['book', 'keyboard', 'apple'], 'data/coco', 'val', image_sq=224, mask_sq=224)

for i in generator.coco_dataset.image_ids:
    image, mask = generator.load_data(i, 224, 224)
    y = model.predict(np.expand_dims(image,axis=0))

    plt.subplot(1, 3, 1)
    plt.imshow((image+1.)/2.)
    plt.title('Scaled image')

    y = np.round(y)
    plt.subplot(1, 3, 2)
    plt.imshow(y[0, :, :, :])
    plt.title('Predicted mask')

    plt.subplot(1, 3, 3)
    plt.imshow(mask[:, :, :].astype(np.float32))
    plt.title('Ground truth')
    plt.show()
