import numpy as np
def normalize_images(images):
    return images.astype('float32') / 255.0


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]



