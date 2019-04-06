# An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions

We present a computationally efficient approach to semantic segmentation, while achieving a high mean intersection over union (mIOU), 70.33% on Cityscapes challenge. The network proposed is capable of running real-time on mobile devices.

Pre-print paper: [https://arxiv.org/abs/1902.07476][4]

If you find the code useful for your research, please consider citing us:

```tex
@article{turkmen2019efficient,
  title={An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions},
  author={T{\"u}rkmen, Sercan and Heikkil{\"a}, Janne},
  journal={arXiv preprint arXiv:1902.07476},
  year={2019}
}
```

## Getting ready

1. Add [`tensorflow/models/slim`][3] to your python path in order to run most of the scripts! (check .env file)
2. Prepare dataset. Example scripts and code is available under the `dataset` folder. The dataset should be in `tfrecord` format.

## Model zoo

| Checkpoint name                         | Eval OS | Eval scales | Left-right Flip |    mIOU     | File Size |
| --------------------------------------- | :-----: | :---------: | :-------------: | :---------: | --------: |
| [shufflenetv2_basic_cityscapes_67_7][1] |   16    |   \[1.0\]   |       No        | 67.7% (val) |     4.9MB |
| [shufflenetv2_dpc_cityscapes_71_3][2]   |   16    |   \[1.0\]   |       No        | 71.3% (val) |     6.3MB |

## Training

To learn more about the available flags you can check `common.py` and the specific script that you are trying to run (e.g. `train.py`).

### Example training configuration

```sh
python train.py \
    --model_variant=shufflenet_v2 \
    --tf_initial_checkpoint=./checkpoints/model.ckpt \
    --training_number_of_steps=120000 \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=True \
    --initialize_last_layer=False \
    --output_stride=16 \
    --train_crop_size=769 \
    --train_crop_size=769 \
    --train_batch_size=16 \
    --dataset=cityscapes \
    --train_split=train \
    --dataset_dir=./dataset/cityscapes/tfrecord \
    --train_logdir=./logs \
    --loss_function=sce
```

### Example evaluation configuration

```sh
python evaluate.py \
    --model_variant=shufflenet_v2 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --output_stride=16 \
    --eval_logdir=./logs/eval \
    --checkpoint_dir=./logs \
    --dataset=cityscapes \
    --dataset_dir=./dataset/cityscapes/tfrecord
```

## Exporting to TFLITE model

`export_tflite.py` script contains several parameters at the top of the script.

## Running on Android

You can find an example script to run the this model and Tensorflow Lite interpreter for segmentation on Android in [this repository][5].

[1]: https://github.com/sercant/mobile-segmentation/releases/download/v0.1.0/shufflenetv2_basic_cityscapes_67_7.zip
[2]: https://github.com/sercant/mobile-segmentation/releases/download/v0.1.0/shufflenetv2_dpc_cityscapes_71_3.zip
[3]: https://github.com/tensorflow/models/tree/master/research/slim
[4]: https://arxiv.org/abs/1902.07476
[5]: https://github.com/sercant/android-segmentation
