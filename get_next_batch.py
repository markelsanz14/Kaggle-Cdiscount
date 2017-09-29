from PIL import Image
import os
import numpy as np
import pandas as pd

#Reads images from directory and creates the training set
def get_next_batch(size, category_dict):
    batch_x = []
    batch_y = []
    path = '../train/'

    random_filenames = random.sample([x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))], size)
    for im in random_filenames:
        image = Image.open(im)
        elems = im.split('-')

        # Add label
        category = elems[len(elems)-1]
        label = category_dict[category]
        label_one_hot = np.zeros(5270)
        label_one_hot[label] = 1
        batch_y.append(label_one_hot)

        # Add features flattened
        image_pixels = np.asarray(image.getdata()).reshape((image.size[1]*image.size[1]*3))
        batch_x.append(image_pixels)


    return batch_x, batch_y
