import pandas as pd
import numpy as np
import create_dict

#vgg = pd.read_csv('vgg_16-result.csv')
inception = pd.read_csv('inception_v4-result.csv')
#resnet = pd.read_csv('resnet_152_v2-result.csv')

cat_to_int, int_to_cat = create_dict.create_dict()

output = []
for index, row in inception.iterrows():
    a = int_to_cat[str(row['category_id'])]
    row['category_id'] = a
    output.append([row['_id'], row['category_id']])
    if index % 1000 == 0:
        print(index)
print(output[:10])
pd.DataFrame(output, columns=['_id', 'category_id']).to_csv('INCEPTION-RESULTT.csv', index=False)
