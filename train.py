import os

import keras
from keras import backend as K
from keras import layers, optimizers, callbacks
import numpy as np

from datasets.coco_dataset import DataGenerator as coco_generator
from nets.mobileunet_v2 import MobileUNet_v2, load_mobilenet_weights
from utils.loss import dice_coef, dice_coef_loss, recall, precision, f1_score

if __name__ == "__main__":
    cat_nms = ['book', 'apple', 'keyboard']

    BATCH_SIZE = 32
    NUM_EPOCH = 250
    IMAGE_SQ_SIZE = 224

    coco_path = '/Volumes/SercanHDD/coco'
    log_path = './logs'
    checkpoint_path = './dist/'

    for path in [coco_path, log_path, checkpoint_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    seed = 1
    np.random.seed(seed)

    training_generator = coco_generator(cat_nms, coco_path, subset='train',
                                        batch_size=BATCH_SIZE, image_sq=IMAGE_SQ_SIZE, mask_sq=int(IMAGE_SQ_SIZE / 2))
    validation_generator = coco_generator(
        cat_nms, coco_path, subset='val', batch_size=BATCH_SIZE, image_sq=IMAGE_SQ_SIZE, mask_sq=int(IMAGE_SQ_SIZE / 2))

    input_tensor = layers.Input(
        shape=(IMAGE_SQ_SIZE, IMAGE_SQ_SIZE, 3), name='input_1')

    net = MobileUNet_v2(
        input_tensor=input_tensor
    )

    model = keras.models.Model(input_tensor, net)

    # Load weights.
    load_mobilenet_weights(model, 1.0, IMAGE_SQ_SIZE)

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(lr=1e-5),
        loss=dice_coef_loss,
        metrics=[dice_coef, recall, precision, f1_score]
    )

    tensorboard = callbacks.TensorBoard(log_dir=log_path)
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path + 'weights.h5',
        save_weights_only=True,
        save_best_only=True)

    # Train model on dataset
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=NUM_EPOCH,
        callbacks=[tensorboard, checkpoint]
    )
