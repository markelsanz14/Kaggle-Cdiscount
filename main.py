from create_dict import create_dict
from conv_net import conv_net
from restore_model import restore_model
import os

# Create dictionary to map category to index
category_to_int, int_to_category = create_dict()

# Call Machine Learning function
if not os.path.exists("../models/model.meta"):
    print('CREATING NEW GRAPH')
    conv_net(category_to_int, int_to_category)
else:
    print('RESTORING EXISTING GRAPH')
    restore_model(category_to_int, int_to_category)
