from PIL import Image
import os
import base64, io
import numpy as np
import pandas as pd
import sqlite3
import json
import matplotlib.pyplot as plt
import random

def get_next_training_batch(batch_size, category_to_int):
    pathW = '/media/markelsanz14/7EA64A44A649FD61/Users/marke/Desktop/myImages.db'
    pathD = '/media/markelsanz14/Markel/myImages.db'
    pathD2 = '/media/markelsanz14/Markel/myImagess.db'
    v = random.random()
    if v < 0.5:
        conn = sqlite3.connect(pathD)
        min_id = 4682600
        max_id = 9682600
    elif v < 0.9:
        conn = sqlite3.connect(pathW)
        min_id = 0
        max_id = 4682600
    else:
        conn = sqlite3.connect(pathD2)
        min_id = 9682600
        max_id = 10682600

    cur = conn.cursor()
    id_list = [x for x in random.sample(range(min_id, max_id), k=batch_size) if x%10 in [0, 1, 3, 4, 5, 6, 8, 9]]
    while len(id_list) < batch_size:
        new_id = random.sample(range(min_id, max_id), k=1)[0]
        if new_id % 10 in [0, 1, 3, 4, 5, 6, 8, 9]:
            id_list.append(new_id)
    images = []
    for id_ in id_list:
        cur.execute("SELECT * FROM Training WHERE id = {}".format(id_))
        images.append(cur.fetchone())
    conn.close()
    batch_x = []
    batch_y = []

    for im in images:
        image = np.fromstring(im[2], dtype="uint8")
        real_image = np.resize(image, (180,180,3))
        elems = im[1].split('-')

        # Add label
        category = elems[len(elems)-1]
        label = int(category_to_int[category])
        label_one_hot = np.zeros(5270)
        label_one_hot[label] = 1
        batch_y.append(label_one_hot)

        # Add features flattened
        image_pixels = np.asarray(image)
        batch_x.append(image_pixels)

    return batch_x, batch_y


def get_next_validation_batch(batch_size, category_to_int):
    pathW = '/media/markelsanz14/7EA64A44A649FD61/Users/marke/Desktop/myImages.db'
    pathD = '/media/markelsanz14/Markel/myImages.db'
    if random.random() > 0.5:
        conn = sqlite3.connect(pathD)
        min_id = 4682600
        max_id = 9682600
    else:
        conn = sqlite3.connect(pathW)
        min_id = 0
        max_id = 4682600
    cur = conn.cursor()
    id_list = [x for x in random.sample(range(min_id, max_id), k=batch_size) if x%10 not in [0, 1, 3, 4, 5, 6, 8, 9]]
    while len(id_list) < batch_size:
        new_id = random.sample(range(min_id, max_id), k=1)[0]
        if new_id % 10 not in [0, 1, 3, 4, 5, 6, 8, 9]:
            id_list.append(new_id)
    images = []
    for id_ in id_list:
        cur.execute("SELECT * FROM Training WHERE id = {}".format(id_))
        images.append(cur.fetchone())
    conn.close()
    batch_x = []
    batch_y = []

    for im in images:
        image = np.fromstring(im[2], dtype="uint8")
        real_image = np.resize(image, (180,180,3))
        elems = im[1].split('-')

        # Add label
        category = elems[len(elems)-1]
        label = int(category_to_int[category])
        label_one_hot = np.zeros(5270)
        label_one_hot[label] = 1
        batch_y.append(label_one_hot)

        # Add features flattened
        image_pixels = np.asarray(image)
        batch_x.append(image_pixels)

    return batch_x, batch_y


