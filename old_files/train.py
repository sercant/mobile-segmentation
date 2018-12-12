import os

from keras import layers, optimizers, callbacks
import numpy as np
import imgaug

from datasets.coco_dataset import DataGenerator as coco_generator
from nets.deeplabv3p import network
from nets.deeplabv3p import load_backbone_weights
from utils.loss import recall, precision, f1_score, bce_dice_loss, jaccard_coef, mean_iou

cat_nms = ['book', 'keyboard', 'apple']

BATCH_SIZE = 32
IMAGE_SQ_SIZE = 224
MASK_SQ_SIZE = 224

coco_path = './data/coco/'
log_path = './logs/'
checkpoint_path = './dist/'


def train(num_epoch, layer_str, checkpoint_file, checkpoint_save_file, learning_rate=0.001):
    for path in [coco_path, log_path, checkpoint_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    seed = 1
    np.random.seed(seed)

    input_tensor = layers.Input(
        shape=(IMAGE_SQ_SIZE, IMAGE_SQ_SIZE, 3), name='input_1')

    model = network(input_tensor=input_tensor, num_classes=len(
        cat_nms), trainable_layers=layer_str)

    # Load weights.
    load_backbone_weights(model, IMAGE_SQ_SIZE)

    if os.path.isfile(checkpoint_file):
        model.load_weights(checkpoint_file)

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(),
        loss=bce_dice_loss,
        metrics=['accuracy', mean_iou, jaccard_coef,
                 recall, precision, f1_score]
    )

    tensorboard = callbacks.TensorBoard(log_dir=log_path)
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_save_file,
        save_weights_only=True,
        save_best_only=True,
        verbose=1)
    es = callbacks.EarlyStopping(patience=10, verbose=1)
    rlrop = callbacks.ReduceLROnPlateau(
        factor=0.1, patience=5, min_lr=0.0001, verbose=1)

    augmentation = imgaug.augmenters.Fliplr(0.5)
    training_generator = coco_generator(cat_nms, coco_path, subset='train',
                                        batch_size=BATCH_SIZE, image_sq=IMAGE_SQ_SIZE, mask_sq=MASK_SQ_SIZE, shuffle=True, augment=augmentation)

    validation_generator = coco_generator(
        cat_nms, coco_path, subset='val', batch_size=BATCH_SIZE, image_sq=IMAGE_SQ_SIZE, mask_sq=MASK_SQ_SIZE, shuffle=False)

    # Train model on dataset
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=num_epoch,
        callbacks=[
            es,
            rlrop,
            tensorboard,
            checkpoint
        ]
    )


if __name__ == "__main__":

    train(50, 'heads', checkpoint_file=checkpoint_path+'/weights_heads.h5',
          checkpoint_save_file=checkpoint_path+'/weights_heads.h5')
    train(100, 'all', checkpoint_file=checkpoint_path+'/weights_heads.h5',
          checkpoint_save_file=checkpoint_path+'/weights_all.h5')
