import pandas as pd
import numpy as np

vgg = pd.read_csv('vgg_16-result.csv')
inception = pd.read_csv('inception_v4-result.csv')
resnet = pd.read_csv('resnet_152_v2-result.csv')

output = []

unique = vgg['product'].unique()
for product in unique:
    vgg_p = vgg[vgg['product'] == product]['category']
    inception_p = inception[inception['product'] == product]['category']
    resnet_p = resnet[resnet['product'] == product]['category']
    if vgg_p == inception_p or vgg_p == resnet_p:
        output.append([product, vgg_p])
    #elif inception_p == resnet_p:
        #output.append([product, inception_p])
    else: # All different
        output.append([product, inception_p]) #ASSIGN BEST

pd.DataFrame(output, columns=['product', 'category']).to_csv('result.csv', index=False)
