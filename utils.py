# coding: utf-8
import os
import numpy as np
from PIL import Image
import tensorflow as tf


def load_images(input_dir, csv_file, index, batch_shape):
    """Images for inception classifier are normalized to be in [-1, 1] interval"""
    images = np.zeros(batch_shape)
    filenames = []
    truelabel = []
    idx = 0
    for i in range(index, min(index + batch_shape[0], 1000)):
        img_obj = csv_file.loc[i]
        ImageID = img_obj['ImageId'] + '.png'
        img_path = os.path.join(input_dir, ImageID)
        images[idx, ...] = np.array(Image.open(img_path)).astype(np.float) / 255.0
        filenames.append(ImageID)
        truelabel.append(img_obj['TrueLabel'])
        idx += 1

    images = images * 2.0 - 1.0
    return images, filenames, truelabel

def save_images(images, filenames, output_dir):
    """Saves images to the output directory."""
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            image = (images[i, :, :, :] + 1.0) * 0.5
            img = Image.fromarray((image * 255).astype('uint8')).convert('RGB')
            img.save(output_dir + filename, quality=95)

def images_to_FD(input_tensor):
    """Process the image to meet the input requirements of FD"""
    ret = tf.image.resize_images(input_tensor, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ret = tf.reverse(ret, axis=[-1])  # RGB to BGR
    ret = tf.transpose(ret, [0, 3, 1, 2])
    return ret