import csv

def create_dict():
    csv_file = csv.reader(open('../category_names.csv'), delimiter=',')
    category_dict = dict()
    for category, val in csv_file:
        category_dict[category] = val
    return category_dict

