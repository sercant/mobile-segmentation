#!/bin/bash
set -e

mkdir COCO;
pushd COCO;

wget http://images.cocodataset.org/zips/train2017.zip;
wget http://images.cocodataset.org/zips/val2017.zip;
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip;

unzip train2017.zip;
unzip val2017.zip;
unzip annotations_trainval2017.zip;

popd;

python build_coco_data.py --dataset_dir ./COCO/ --output_dir ./COCO/tfrecord --category_names person,car,truck,bus,train,motorcycle,bicycle,stop\ sign,parking\ meter --min_pixels 1000;
