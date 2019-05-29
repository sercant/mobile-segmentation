# An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions

We present a computationally efficient approach to semantic segmentation, while achieving a high mean intersection over union (mIOU), 70.33% on Cityscapes challenge. The network proposed is capable of running real-time on mobile devices.

Paper: [10.1007/978-3-030-20205-7_4][4]

If you find the code useful for your research, please consider citing us:

```tex
@InProceedings{turkmen2019efficient,
  author    = {Sercan T{\"u}rkmen and Janne Heikkil{\"a}},
  title     = {An Efficient Solution for Semantic Segmentation: {ShuffleNet} V2 with Atrous Separable Convolutions},
  booktitle = {Image Analysis},
  year      = {2019},
  editor    = {Michael Felsberg and Per-Erik Forss{\'e}n and Ida-Maria Sintorn and Jonas Unger},
  volume    = {11482},
  pages     = {41--53},
  address   = {Cham},
  publisher = {Springer International Publishing},
  doi       = {10.1007/978-3-030-20205-7_4},
  isbn      = {978-3-030-20205-7},
  url       = {http://dx.doi.org/10.1007/978-3-030-20205-7_4},
}
```

## Getting ready

1. Add [`tensorflow/models/slim`][3] to your python path in order to run most of the scripts! To do so follow these steps:
   1. Clone or download the [`tensorflow/models/slim`][3] repository to a separate folder.
   2. Add the path to the repository by running the following code: `export PYTHONPATH=path_to_the_cloned_folder/tensorflow_models/research/slim:${PYTHONPATH}`
2. Prepare dataset. Example scripts and code is available under the `dataset` folder. The dataset should be in `tfrecord` format.

## Model zoo

| Checkpoint name                         | Trained on                                          | Uses DPC | Eval OS | Eval scales | Left-right Flip |    mIOU     | File Size |
| --------------------------------------- | --------------------------------------------------- | :--: | :-----: | :---------: | :-------------: | :---------: | --------: |
| [shufflenetv2_basic_cityscapes_67_7][1] | MS COCO 2017* + Cityscapes coarse + Cityscapes fine | No |  16    |   \[1.0\]   |       No        | 67.7% (val) |     4.9MB |
| [shufflenetv2_dpc_cityscapes_71_3][2]   | MS COCO 2017* + Cityscapes coarse + Cityscapes fine | Yes |  16    |   \[1.0\]   |       No        | 71.3% (val) |     6.3MB |

\* Filtered to include only `person`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `stop sign`, `parking meter` classes and samples that contain over 1000 annotated pixels.

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

**Important:** To use DPC architecture in your model, you should also set this parameter:

    --dense_prediction_cell_json=./core/dense_prediction_cell_branch5_top1_cityscapes.json

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

**Important:** If you are trying to evaluate a checkpoint that uses DPC architecture, you should also set this parameter:  

    --dense_prediction_cell_json=./core/dense_prediction_cell_branch5_top1_cityscapes.json

## Exporting to TFLITE model

`export_tflite.py` script contains several parameters at the top of the script.

## Running on Android

You can find an example script to run the this model and Tensorflow Lite interpreter for segmentation on Android in [this repository][5].

[1]: https://github.com/sercant/mobile-segmentation/releases/download/v0.1.0/shufflenetv2_basic_cityscapes_67_7.zip
[2]: https://github.com/sercant/mobile-segmentation/releases/download/v0.1.0/shufflenetv2_dpc_cityscapes_71_3.zip
[3]: https://github.com/tensorflow/models/tree/v1.13.0/research/slim
[4]: https://doi.org/10.1007/978-3-030-20205-7_4
[5]: https://github.com/sercant/android-segmentation
