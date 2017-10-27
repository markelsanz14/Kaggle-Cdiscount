import csv

def create_dict():
    csv_file = csv.reader(open('../category_names.csv'), delimiter=',')
    category_to_int = dict()
    int_to_category = dict()
    for category, val in csv_file:
        category_to_int[category] = val
        int_to_category[val] = category
    return category_to_int, int_to_category

