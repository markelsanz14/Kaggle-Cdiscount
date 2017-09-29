import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from scipy.ndimage import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
import pandas as pd

# Simple data processing

data = bson.decode_file_iter(open('../train_example.bson', 'rb'))

for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    picture_id = 0
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        # do something with the picture, etc
        plt.imsave('../train/'+str(product_id)+'-'+str(picture_id)+'-'+str(category_id)+'.png', picture)
        picture_id += 1

