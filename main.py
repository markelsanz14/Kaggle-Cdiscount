from create_dict import create_dict
from conv_net import conv_net

# Create dictionary to map category to index
category_to_int, int_to_category = create_dict()

# Call Machine Learning function
conv_net(category_to_int, int_to_category)

