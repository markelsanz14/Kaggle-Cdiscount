import glob
import os
import sqlite3

path1 = '../train2/'
def genimages():
    """Generate example images."""
    print('genimages called')
    i = 0
    for pngpath in glob.iglob(path1+'*.png'):
        with open(pngpath, 'rb') as f:
            name = pngpath.split('/')[len(pngpath)-1]
            if i % 100 == 0:
                print(i)
            i += 1
            yield i, name, f.read()


try:
    connection = sqlite3.connect('../myImages.db')
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS Training(id int, name string, image blob)")
    print('DB created')
    cursor.executemany("INSERT INTO Training(name,image) values(?,?)", genimages())
    print('DB fully populated')
    connection.commit()
finally:
    connection.close()

