import sqlite3
import os
import pandas as pd
import numpy as np
import re
from time import process_time
from itertools import islice

# Create DB
#os.remove("MultiDark.db")
schema = "./schema.txt"
TableName = "MultiDark"
with open(schema, 'r') as myfile:
  data = myfile.read()
conn = sqlite3.connect("MultiDark.db")
c = conn.cursor()
createTableString = "CREATE TABLE " + TableName + " (" + data + ")"
#c.execute(createTableString)
print(createTableString)

headernames = "./headernames.txt"
with open(headernames, 'r') as headerfile:
    header_data = headerfile.read()
header_data = re.sub("\n", "", header_data)
header_list = np.array(header_data.split(", "))

fields_string = "("
for element in header_list:
    fields_string += "'" + element + "',"
fields_string = fields_string[:-1] + ")"

def utf8len(s):
    return len(s.encode('utf-8'))

def insertintoDB(filename):
    size_bytes = os.path.getsize(filename)
    chunks = 0
    size_bool = False

    chunksize = 8 #5000
    for chunk in pd.read_csv(filename, chunksize=chunksize, header = None, delimiter = "\s+", comment = '#'):
        t1_start = process_time()

        a = chunk.to_numpy()

        print("Values: ", a)
        break
        a = np.array2string(a, separator = ',' )

        a = re.sub("\]", ")", a)
        a = re.sub("\)\)", ")", a)

        a = re.sub("\[", "(", a)
        a = re.sub("\(\(", "(", a)

        #a = chunk.to_string(index = False, header = False)

        #a = re.sub("\n", "),\n(", a)

        #a = "(" + re.sub("[ \t]+", ", ", a) + ");"

        chunks += 1
        full_string =  "INSERT INTO " + TableName + " " + fields_string + " VALUES\n"

        if not size_bool:
            row = chunk.iloc[[0]].to_string(index = False, header = False)
            size_string = utf8len(row)
            size_bool = True

        full_string += a + ";"
        print(full_string)

        c.execute(full_string)

        t1_stop = process_time()
        print("Submission:", t1_stop-t1_start)

        break

        percentage = (chunks * chunksize * size_string)/size_bytes * 100
        print("{} - Percentage Complete: ".format(filename), percentage)

def insertintoDB_string(filename):
    t1_start = process_time()
    size_bool = False
    size_bytes = os.path.getsize(filename)
    chunks = 0

    n = 10000  # Or whatever chunk size you want
    with open(filename, 'rb') as f:
        batch = ""
        count = 0
        for line in f:
            decoded = line.decode("utf-8")
            if decoded[0] == "#":
                pass;
            else:
                if not size_bool:
                    line_size = utf8len(decoded)
                    size_bool = True
                batch += decoded
                count += 1
                if count == n:
                    chunks += 1
                    batch = re.sub(" +", ", ", batch)
                    batch = re.sub("\n", "),\n(", batch)
                    batch = batch[:-3]

                    full_string =  "INSERT INTO "\
                                    + TableName\
                                    + " "\
                                    + fields_string\
                                    + " VALUES\n"\
                                    + "(" + batch

                    c.execute(full_string)
                    conn.commit()

                    percentage = (chunks * n * line_size)/size_bytes * 100
                    print("Percentage: {}%".format(filename), percentage, end = '\r')

                    batch = ""
                    count = 0

#string_list = ["hlist_1.00000.list", "hlist_0.91520.list", "hlist_0.80130.list",
#               "hlist_0.65650.list", "hlist_0.50320.list", "hlist_0.40320.list",  "hlist_0.33030.list"]

string_list = ["hlist_0.91520.list"]



for element in string_list:

    t1_start = process_time()
    insertintoDB_string(element)
    t1_stop = process_time()
    print("Time: ", (t1_stop-t1_start)/(60*60), " hours")


conn.commit()
conn.close()
