import argparse
from vectordb.benchmarks import get_mult_true_neighbours
from vectordb.indexing import get_vectors_with_ids, sep_results, list_to_list
import csv
import pandas as pd
import numpy as np
import multiprocessing
from lithops import Storage

## General arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", default=12, help="Number of workers for numpy")
parser.add_argument("--features", default=100, help="Number of features for each vector")    
parser.add_argument("--k_result", default=10, help="Number of neighbours to search for")
parser.add_argument("--dataset", default="glove", help="Name of the dataset to be used")
parser.add_argument("--storage_bucket", default="xroca-vectordb", help="Storage bucket name")

## Get all arguments
args = parser.parse_args()

# Read queries
print("Read queries")
query_vectors = []
with open(f'queries_{args.dataset}.csv', mode ='r')as file:
    csvFile = csv.reader(file)
        
    for lines in csvFile:
        vector = lines[0].split(" ")
        vector = [float(value) for value in vector if value != '']
        query_vectors.append(vector)
        
#Read vectors
print("Read vectors")
df = pd.read_csv(f"vectors_{args.dataset}.csv", header=None)
df = np.array_split(df, int(args.num_workers))
pool = multiprocessing.Pool(processes=int(args.num_workers))
res = pool.map(get_vectors_with_ids, df)
pool.close()
    
ids, vectors = sep_results(res)
ids = list_to_list(ids)
vectors = list_to_list(vectors)

# Perform search
print("Perform search")

query_vectors_1k = query_vectors[:1000]

true = get_mult_true_neighbours(query_vectors[:1000], vectors, ids, int(args.features), int(args.k_result))

#Upload results
print("Upload results")
res = []
for x in true:
    res.append(x[0])
    
df = pd.DataFrame(res)

storage = Storage()
storage.put_object(bucket=args.storage_bucket, key=f"true_neighbours_{args.dataset}.csv", body=df.to_csv(index=False, header=False))