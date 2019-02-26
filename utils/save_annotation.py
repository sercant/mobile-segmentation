
"""Saves an annotation as one png image.

This script saves an annotation as one png image, and has the option to add
colormap to the png image for better visualization.
"""

import numpy as np
import PIL.Image as img
import tensorflow as tf

from utils import get_dataset_colormap


def save_annotation(label,
                    save_dir,
                    filename,
                    add_colormap=True,
                    colormap_type=get_dataset_colormap.get_pascal_name()):
    """Saves the given label to image on disk.

    Args:
      label: The numpy array to be saved. The data will be converted
        to uint8 and saved as png image.
      save_dir: The directory to which the results will be saved.
      filename: The image filename.
      add_colormap: Add color map to the label or not.
      colormap_type: Colormap type for visualization.
    """
    # Add colormap for visualizing the prediction.
    if add_colormap:
        colored_label = get_dataset_colormap.label_to_color_image(
            label, colormap_type)
    else:
        colored_label = label

    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    if not isinstance(filename, str):
        filename = filename.decode('utf-8')
    with tf.gfile.Open("{}/{}.png".format(save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')
