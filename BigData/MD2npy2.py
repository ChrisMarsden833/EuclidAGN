import os
from sys import argv
import numpy as np
import pandas as pd
import csv
from sys import getsizeof

from time import time, sleep
from datetime import datetime, timedelta

files = ["hlist_0.33030.list", "hlist_0.40320.list", "hlist_0.50320.list", "hlist_0.65650.list", "hlist_0.80130.list", "hlist_0.91520.list", "hlist_1.00000.list"]

supertime = time()

for fname_input in files:

    print("File", fname_input)

    def utf8len(s):
        return len(s.encode('utf-8'))

    nowtime = time()

    with open('header.txt', 'r') as f:
        reader = csv.reader(f)
        names = list(reader)[0]
    with open('dtypes.txt', 'r') as f:
        reader = csv.reader(f)
        types = list(reader)[0]

    input_types = {}
    for i, name in enumerate(names):
        input_types[name] = types[i]

    hlist = ["x", "y", "z", "id", "upid", "mvir", "Acc_Scale", "Macc"]

    type =  [("x", np.float32),
             ("y", np.float32),
             ("z", np.float32),
             ("id", np.int64),
             ("upid", np.int64),
             ("mvir", np.float32),
             ("Acc_Scale", np.float32),
             ("Macc", np.float32)]

    first = True

    dat = np.array([], dtype = type)

    chunks = 0
    chunksize = 10**6

    size_bytes = os.path.getsize(fname_input)

    for df in pd.read_csv(fname_input, delimiter = "\s+", chunksize = chunksize, comment = '#', names = names, dtype = input_types):



        data = df[hlist].to_records(index = False) #to_numpy(dtype = type, copy = True)

        if chunks == 0:
            dat = data.copy()
        else:
            dat = np.concatenate((dat, data))

        #print(dat.dtype)

        if first:
            scale = df["scale"][0]
            redshift = 1/scale - 1

            row = df.iloc[[0]].to_string(index = False, header = False)
            size_string = utf8len(row)
            #row = np.array2string(df.values[0, :])
            size_string = utf8len(row) #- (12*4)
            print("Bytes:", size_string)
            first = False

        #print(dat.dtype)

        chunks += 1
        percentage = (chunks * chunksize * size_string)/size_bytes * 100
        si = getsizeof(dat)/10**9
        print('{} - Percentage Complete: {}%, data size: {}GB \r'.format(fname_input, np.round(percentage, 2), np.round(si, 3)), end="")

    #print(data.values[:, 0])

    fname_output = "MD_{}.npy".format(np.round(redshift, 1))

    print("Saving file", fname_output)
    #dat = dat.view(type)
    np.save(fname_output, dat)

    sec = timedelta(seconds=int(time()-nowtime))
    d = datetime(1,1,1)+sec

    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))

sec = timedelta(seconds=int(time()-supertime))
d = datetime(1,1,1)+sec
print("Overall:")
print("DAYS:HOURS:MIN:SEC")
print("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))
