
import os
import shutil
import zipfile
import urllib.request
import warnings

import keras
from keras.applications import imagenet_utils
import numpy as np
import skimage
import scipy
from mrcnn import utils

from datasets.coco.coco import COCO
from datasets.coco import mask as maskUtils

DEFAULT_DATASET_YEAR = "2017"

resize_image = utils.resize_image
resize_mask = utils.resize_mask

# def resize_image(image, min_dim=224):
#     # Keep track of image dtype and return results in the same dtype
#     image_dtype = image.dtype
#     # Default window (y1, x1, y2, x2) and default scale == 1.
#     h, w = image.shape[:2]
#     window = (0, 0, h, w)
#     scale = 1
#     padding = [(0, 0), (0, 0), (0, 0)]
#     crop = None

#     # Scale?
#     if min_dim:
#         scale = min_dim / min(h, w)

#     # Resize image using bilinear interpolation
#     if scale != 1:
#         image = skimage.transform.resize(
#             image, (round(h * scale), round(w * scale)),
#             order=1, mode="constant", preserve_range=True) # , anti_aliasing=scale<1.)

#     # Need padding or cropping?
#     # Pick a random crop
#     h, w = image.shape[:2]
#     if w != min_dim or h != min_dim:
#         y = np.random.randint(0, (h - min_dim)) if h != min_dim else 0
#         x = np.random.randint(0, (w - min_dim)) if w != min_dim else 0
#         crop = (y, x, min_dim, min_dim)
#         image = image[y:y + min_dim, x:x + min_dim]
#         window = (0, 0, min_dim, min_dim)
#     return image.astype(image_dtype), window, scale, padding, crop


# def resize_mask(mask, scale, padding, crop=None):
#     """Resizes a mask using the given scale and padding.
#     Typically, you get the scale and padding from resize_image() to
#     ensure both, the image and the mask, are resized consistently.

#     scale: mask scaling factor
#     padding: Padding to add to the mask in the form
#             [(top, bottom), (left, right), (0, 0)]
#     """
#     # Suppress warning from scipy 0.13.0, the output shape of zoom() is
#     # calculated with round() instead of int()
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
#     if crop is not None:
#         y, x, h, w = crop
#         mask = mask[y:y + h, x:x + w]
#     else:
#         mask = np.pad(mask, padding, mode='constant', constant_values=0)
#     return mask

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, cat_nms=None, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO(
            "{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        if cat_nms:
            class_ids = coco.getCatIds(catNms=cat_nms)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for _id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[_id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(
                "val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(
                dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(
                dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(
                dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(
                annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(
                dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds - deleted @sercant
                    # class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones(
                            [image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, cat_nms, path='./data/coco', subset='train', batch_size=32, image_sq=224, mask_sq=112, shuffle=True, n_channels=3, augment=None):
        'Initialization'
        self.image_sq = image_sq
        self.mask_sq = mask_sq
        self.batch_size = batch_size
        self.cat_nms = cat_nms
        self.subset = subset
        self.n_classes = len(cat_nms)
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.augment = augment

        coco_dataset = CocoDataset()
        coco_dataset.load_coco(path, subset, year='2017',
                               auto_download=True, cat_nms=cat_nms)
        coco_dataset.prepare()

        self.coco_dataset = coco_dataset
        self.image_ids = coco_dataset.image_ids

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        image_ids_temp = [self.image_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def mask_to_one_hot(self, mask, class_ids):
        output = np.zeros([self.n_classes, mask.shape[0],
                           mask.shape[1]], dtype=np.float32)

        mask = np.moveaxis(mask, -1, 0)
        for i in range(len(mask)):
            cl = class_ids[i] - 1  # substract bg class
            assert 0 <= cl < self.n_classes
            output[cl] = np.logical_or(output[cl], mask[i])

        return np.moveaxis(output, 0, -1).astype(np.uint8)

    def augmentation(self, image, mask, augmentation):
        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask, hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

        return image, mask

    def load_data(self, image_id, image_sq, mask_sq):
        image = self.coco_dataset.load_image(image_id)
        mask, class_ids = self.coco_dataset.load_mask(image_id)

        image, _, scale, padding, crop = resize_image(
            image, min_dim=image_sq, max_dim=image_sq)
        mask = resize_mask(mask, scale, padding, crop)

        if self.augment:
            image, mask = self.augmentation(image, mask, self.augment)

        image = imagenet_utils.preprocess_input(image, mode='tf')
        # image = image / 128. - 1.

        if image_sq != mask_sq:
            mask = resize_mask(mask, float(
                mask_sq) / image_sq, [(0, 0), (0, 0), (0, 0)])
        mask = self.mask_to_one_hot(mask, class_ids)

        return image, mask

    def __data_generation(self, image_ids_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty(
            (self.batch_size, self.image_sq, self.image_sq, self.n_channels), dtype=np.float)
        y = np.empty(
            (self.batch_size, self.mask_sq, self.mask_sq, self.n_classes), dtype=np.uint8)

        # Generate data
        for i, ID in enumerate(image_ids_temp):
            image, mask = self.load_data(ID, self.image_sq, self.mask_sq)

            X[i, ] = image
            y[i, ] = mask

        return X, y


if __name__ == '__main__':
    dataset_train = CocoDataset()
    dataset_train.load_coco('./data/coco', "train", year=DEFAULT_DATASET_YEAR,
                            auto_download=True, cat_nms=['book', 'apple', 'keyboard'])

    dataset_train.prepare()
