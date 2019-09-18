import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset
from preprocess import preprocess
from model import shufflenet_v2_segmentation
from utils.losses import SoftmaxCrossEntropy
from utils.metrics import IgnoreLabeledMeanIoU

dataset_name = "pascal_voc_seg"
dataset_dir = "data/pascal/tfrecord/"
dataset_split = "train"
batch_size = 12
input_size = [512, 512]

dataset, dataset_desc = get_dataset(dataset_name, dataset_split, dataset_dir)
preprocessed_dataset = dataset.map(
    preprocess(
        input_size, is_training=True,
        ignore_label=dataset_desc.ignore_label)).shuffle(512).batch(batch_size)

val_dataset, _ = get_dataset(dataset_name, 'val', dataset_dir)
preprocessed_val_dataset = val_dataset.map(
    preprocess(input_size,
               is_training=False,
               ignore_label=dataset_desc.ignore_label)).batch(32)

inputs = keras.Input(shape=(input_size[0], input_size[1], 3))
outputs = shufflenet_v2_segmentation(
    inputs,
    dataset_desc.num_classes,
    feature_extractor_checkpoint='./checkpoints/cifar10.hdf5')
model = keras.Model(inputs=inputs, outputs=outputs)

train_loss = SoftmaxCrossEntropy(dataset_desc.num_classes,
                                 dataset_desc.ignore_label)
train_accuracy = IgnoreLabeledMeanIoU(dataset_desc.num_classes,
                                      dataset_desc.ignore_label)

learning_rate_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.1)
model_checkpoint = keras.callbacks.ModelCheckpoint('model.h5',
                                                   save_best_only=True,
                                                   verbose=True)

optimizer = tf.keras.optimizers.Adam()

model.compile(loss=train_loss, optimizer=optimizer, metrics=[train_accuracy])
model.fit_generator(preprocessed_dataset,
                    epochs=50,
                    validation_data=preprocessed_val_dataset,
                    callbacks=[learning_rate_scheduler, model_checkpoint])
