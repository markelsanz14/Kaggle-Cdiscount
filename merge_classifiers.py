import pandas as pd
import numpy as np

vgg = pd.read_csv('VGG-RESULTT.csv')
inception = pd.read_csv('INCEPTION-RESULTT.csv')
resnet = pd.read_csv('RESNET-RESULTT.csv')

output = []

for i in range(len(vgg)):
    if inception.iloc[i]['category_id'] == resnet.iloc[i]['category_id'] or vgg.iloc[i]['category_id'] == inception.iloc[i]['category_id']:
        output.append([inception.iloc[i]['_id'], inception.iloc[i]['category_id']])
    elif vgg.iloc[i]['category_id'] == resnet.iloc[i]['category_id']:
        output.append([vgg.iloc[i]['_id'], vgg.iloc[i]['category_id']])
    else:
        output.append([resnet.iloc[i]['_id'], resnet.iloc[i]['category_id']])

    if i % 1000 == 0:
        print(i)

pd.DataFrame(output, columns=['_id', 'category_id']).to_csv('RESULT.csv', index=False)
