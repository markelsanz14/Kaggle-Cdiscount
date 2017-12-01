import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from scipy.ndimage import imread   # or, whatever image library you prefer
import pandas as pd
import os
import sqlite3
import numpy as np
import tensorflow as tf

from create_dict import create_dict

#Wrapper functions used to convert data into tfrecord data
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


category_to_int, int_to_category = create_dict()

print(category_to_int)
print(category_to_int['1000010653'])

train_record = '/home2/kvtran2/CDiscount/cdiscount_train.tfrecord'
#validate_record = 'D:/CDiscount/cdiscount_validation.tfrecord'
train_writer = tf.python_io.TFRecordWriter(train_record)
#validate_writer = tf.python_io.TFRecordWriter(validate_record)

data = bson.decode_file_iter(open('../train.bson', 'rb'))

i = 0

for c, d in enumerate(data):
    #if i > 10000000:
    #break
    product_id = d['_id']
    category_id = d['category_id']
    picture_id = 0
    for e, pic in enumerate(d['imgs']):
        if i > -1:
            image_buffer = pic['picture']
            picture = imread(io.BytesIO(pic['picture']))

            colorspace = "RGB"
            channels = 3
            image_format = 'JPEG'
            label = int(category_to_int[str(d['category_id'])])
            text = str(d['category_id']).encode('utf8')

            name = str(product_id) + '-' + str(picture_id) + '-' + str(category_id)

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': _int64_feature(picture.shape[0]),
                'image/width': _int64_feature(picture.shape[1]),
                'image/colorspace': _bytes_feature(colorspace.encode('utf8')),
                'image/channels': _int64_feature(channels),
                'image/class/label': _int64_feature(label),
                'image/class/text': _bytes_feature(text),
                'image/format': _bytes_feature(image_format.encode('utf8')),
                'image/filename': _bytes_feature(os.path.basename(name).encode('utf8')),
                'image/encoded': _bytes_feature(image_buffer)}))

            if i % 10000 == 0:
                print(i)

            train_writer.write(example.SerializeToString())
            picture_id += 1
            i += 1
        else:
            i += 1
            if i % 10000 == 0:
                print(i)

train_writer.close()
#validate_writer.close()
