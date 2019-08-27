import os
from sys import argv
import numpy as np
import pandas as pd
import csv

from time import time, sleep
from datetime import datetime, timedelta

def utf8len(s):
    return len(s.encode('utf-8'))

nowtime = time()

with open('header.txt', 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)[0]

list = ["x", "y", "z", "Acc_Scale", "Macc", "mvir"]

fname_input = argv[1]

first = True

dat = []

chunks = 0
chunksize = 10**7

size_bytes = os.path.getsize(fname_input)

for df in pd.read_csv(fname_input, delimiter = "\s+", names = your_list, chunksize = chunksize, comment = '#'):


    data = df[list].values

    #print(data)

    if chunks == 0:
        dat = data.copy()
    else:
        dat = np.concatenate((dat, data))


    if first:
        scale = df["scale"][0]
        redshift = 1/scale - 1

        row = np.array2string(data[0, :])
        size_string = utf8len(row)
        first = False


    chunks += 1
    percentage = (chunks * chunksize * size_string)/size_bytes * 100
    print('{} - Percentage Complete: {}%\r'.format(fname_input, np.round(percentage, 2)), end="")


#print(data.values[:, 0])

fname_output = "MD_{}.npy".format(np.round(redshift, 1))

np.save(fname_output, data)

sec = timedelta(seconds=int(time()-nowtime))
d = datetime(1,1,1)+sec

print("DAYS:HOURS:MIN:SEC")
print("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))
