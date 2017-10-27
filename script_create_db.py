import glob
import os
import sqlite3

path1 = '../train2/'
def genimages():
    """Generate example images."""
    i = 0
    for pngpath in glob.iglob(path1+'*.png'):
        with open(pngpath, 'rb') as f:
            name = pngpath.split('/')[len(pngpath)-1]
            if i % 100 == 0:
                print(i)
            i += 1
            yield i, name, buffer(f.read())


try:
    connection = sqlite3.connect('../myImages.db')
    cursor = connection.cursor()
    cursor.execute("CREATe TABLE IF NOT EXISTS Training(id int, name string, image blob)")
    cursor.executemany("INSERT INTO Training(name,image) values(?,?)", genimages())

    connection.commit()
finally:
    connection.close()

