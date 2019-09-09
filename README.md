# An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions

We present a computationally efficient approach to semantic segmentation, while achieving a high mean intersection over union (mIOU), 70.33% on Cityscapes challenge. The network proposed is capable of running real-time on mobile devices.

Published paper: [https://doi.org/10.1007/978-3-030-20205-7_4][4]

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

## Work in Progress

This is the re-written version of the network using Tensorflow 2.0-rc0 and Keras. It is still incomplete and the checkpoints are missing.

[1]: https://github.com/sercant/mobile-segmentation/releases/download/v0.1.0/shufflenetv2_basic_cityscapes_67_7.zip
[2]: https://github.com/sercant/mobile-segmentation/releases/download/v0.1.0/shufflenetv2_dpc_cityscapes_71_3.zip
[3]: https://github.com/tensorflow/models/tree/v1.13.0/research/slim
[4]: https://doi.org/10.1007/978-3-030-20205-7_4
[5]: https://github.com/sercant/android-segmentation
