import pandas as pd
import numpy as np

#names = ['vgg_16', 'inception_v4', 'resnet_152_v2']
names = ['data']

for name in names:
    f = pd.read_csv('{}.csv'.format(name))

    output = []
    unique = f['product'].unique()
    for product in unique:
        same_product = f[f['product'] == product]
        unique_categories = same_product['class'].unique()
        probs = {}
        for category in unique_categories:
            same_category = same_product[same_product['class'] == category]
            for index, row in same_category.iterrows():
                if row['class'] in probs.keys():
                    probs[int(row['class'])] += float(row['prob'])
                else:
                    probs[int(row['class'])] = float(row['prob'])
        keys = list(probs.keys())
        values = list(probs.values())
        max_pos = np.argmax(values)
        max_key = keys[max_pos]
        output.append([product, max_key])

    pd.DataFrame(output, columns=['product', 'category']).to_csv('{}-result.csv'.format(name), index=False)
