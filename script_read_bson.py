import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from scipy.ndimage import imread   # or, whatever image library you prefer
import pandas as pd
import glob
import os
import sqlite3
import numpy as np
import json

def genimages():
    data = bson.decode_file_iter(open('../train.bson', 'rb'))
    print('genimages called')
    i=0
    for c, d in enumerate(data):
        if i > 200:
            break
        product_id = d['_id']
        category_id = d['category_id'] # This won't be in Test data
        picture_id = 0
        for e, pic in enumerate(d['imgs']):
            if i > -1:
                picture = imread(io.BytesIO(pic['picture']))
                print(picture.shape)
                picture = picture.tostring()
                print(len(picture))
                name = str(product_id)+'-'+str(picture_id)+'-'+str(category_id)
                picture_id += 1
                i += 1
                if i % 200 == 0:
                    print(i)
                yield i, name, picture 
            else:
                i += 1
                if i % 1000 == 0:
                    print(i)

try:
    pathW = 'D:/CDiscount/myImages.db'
    pathM = '/media/markelsanz14/Markel/myImages.db'
    connection = sqlite3.connect(pathW)
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS Training(id int, name string, image blob)")
    print('DB created')
    cursor.executemany("INSERT INTO Training(id, name, image) values(?,?,?)", genimages())
    connection.commit()
    print('DB correctly populated')
finally:
    connection.close()

