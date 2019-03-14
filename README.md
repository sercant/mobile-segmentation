# An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions

We present a computationally efficient approach to semantic segmentation, meanwhile achieving a high mIOU, 70.33% on Cityscapes challenge. The network proposed is capable of running real-time on mobile devices. In addition, we make our code and model weights publicly available.

## Training

You should add tensorflow/models/slim to your python path in order to run most of the files! (check .env file)

### Preparing the dataset

Examples for preparing the database are under `dataset` folder.

### Example training configuration

```sh
python train.py \
    --model_variant=shufflenet_v2 \
    --tf_initial_checkpoint=./checkpoints/shufflenet_v2_imagenet/model.ckpt-1661328 \
    --training_number_of_steps=120000 \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=True \
    --initialize_last_layer=False \
    --output_stride=16 \
    --train_crop_size=769 \
    --train_crop_size=769 \
    --train_batch_size=16 \
    --dataset=coco \
    --train_split=train \
    --dataset_dir=./dataset/COCO/tfrecord \
    --train_logdir=./logs \
    --loss_function=sce
```

### Example evaluation configuration

```sh
python evaluate.py \
    --model_variant=shufflenet_v2 \
    --eval_crop_size=769 \
    --eval_crop_size=769 \
    --min_resize_value=768 \
    --max_resize_value=768 \
    --scale_factor=16 \
    --output_stride=16 \
    --eval_logdir=./logs/eval \
    --checkpoint_dir=./logs \
    --dataset=coco \
    --dataset_dir=./dataset/COCO/tfrecord
```

## Exporting TFLITE model

`export_tflite.py` script contains several parameters at the top of the script.
