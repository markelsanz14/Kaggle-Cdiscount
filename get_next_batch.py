from PIL import Image
import os
import numpy as np
import pandas as pd

def create_connection():
    try:
        conn = sqlite3.connect("../myImages.db")
        return conn
    except Error as e:
        print(e)
 
    return None


def get_next_training_batch(batch_size, category_to_int):
    conn = create_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM Training ORDER BY RANDOM() WHERE id % 10 BETWEEN 0 AND 7 LIMIT {}".format(batch_size))
        images = cur.fetchall()
        batch_x = []
        batch_y = []

        for im in images:
            #image = Image.open(im)
            image = im[2]
            elems = im[1].split('-')

            # Add label
            category = elems[len(elems)-1]
            label = category_to_int[category]
            label_one_hot = np.zeros(5270)
            label_one_hot[label] = 1
            batch_y.append(label_one_hot)

            # Add features flattened
            image_pixels = np.asarray(image.getdata()).reshape((image.size[1]*image.size[1]*3))
            batch_x.append(image_pixels)


    return batch_x, batch_y
    

def get_next_validation_batch(batch_size, category_to_int):
    conn = create_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM Training ORDER BY RANDOM() WHERE id % 10 BETWEEN 8 AND 9 LIMIT {}".format(batch_size))
        images = cur.fetchall()
        batch_x = []
        batch_y = []

        for im in images:
            #image = Image.open(im)
            image = im[2]
            elems = im[1].split('-')

            # Add label
            category = elems[len(elems)-1]
            label = category_to_int[category]
            label_one_hot = np.zeros(5270)
            label_one_hot[label] = 1
            batch_y.append(label_one_hot)

            # Add features flattened
            image_pixels = np.asarray(image.getdata()).reshape((image.size[1]*image.size[1]*3))
            batch_x.append(image_pixels)


    return batch_x, batch_y


