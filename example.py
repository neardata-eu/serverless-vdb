import argparse
import numpy as np
from vectordb.benchmarks import calculate_mult_recall
from vectordb.serverless_vectordb import ServerlessVectorDB
import csv
import json
from lithops import Storage
from io import StringIO
import time

def calculate_mean(data):
   
    aux = 0
    for x in data:
        aux += x
    return aux / len(data)

def calculate_mean_mult(data):
    aux = 0
    total_len = 0
    for x in data:
        total_len += len(x)
        for y in x:
            aux += y
            
    return aux / total_len

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


if __name__ == "__main__":
    ## General arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=100, help="Number of features for each vector")
    parser.add_argument("--num_vectors", default=-1, help="Number of vectors in the dataset. Only needed on centroids implementation")
    parser.add_argument("--k_search", default=10, help="Number of neighbours to search for. Should be the same as k_result")    
    parser.add_argument("--k_result", default=10, help="Number of neighbours to search for. Should be the same as k_search")
    parser.add_argument("--skip_init", action="store_true", default=False, help="Skip vector database initialization")
    parser.add_argument("--skip_query", action="store_true", default=False, help="Skip vector database querying")
    parser.add_argument("--skip_global_kmeans", action="store_true", default=False, help="Skip global balanced K-means to find centroids")
    parser.add_argument("--impl", default="blocks", help="Implementation: blocks or centroids")
    parser.add_argument("--kmeans_version", default="unbalanced", help="Kmeans implementation: unbalanced (faiss Kmeans) or balanced (k_means_constrained)")
    parser.add_argument("--dataset", default="glove", help="Name of the dataset to be used")
    
    ## Custom algorithm arguments
    parser.add_argument("--replication_percentage", default=0, help="Percentage of top nearest centroids for vector replication when assigning vectors to their corresponding centroid on centroids implementation.")
    parser.add_argument("--num_index", default=16, help="Number of centroids to divide the vectors into")
    parser.add_argument("--num_centroids_search", default=4, help="Number of indexes to search for in the custom algorithm")
    parser.add_argument("--k", default=4096, help="Number of clusters to create within each partition in the custom algorithm")
    parser.add_argument("--n_probe", default=1024, help="Number of clusters to search for in the custom algorithm")
    parser.add_argument("--query_batch_size", default=16, help="Number of indexes to be queried per function. Combined with the num_index, it determines the amount of functions to be raised on querying")
    
    ## Storage arguments
    parser.add_argument("--storage_bucket", default="xroca-vectordb", help="Storage bucket name")
    
    ## Runtime arguments
    parser.add_argument("--indexing_memory", default=8192, help="Memory of the indexing map functions")
    parser.add_argument("--search_map_memory", default=9216, help="Memory of the get_mult_neighbours search functions")
    parser.add_argument("--search_reduce_memory", default=2048, help="Memory of the reduce_mult_neighbours search functions")

    ## Get all arguments
    args = parser.parse_args()

    sv_vectordb = ServerlessVectorDB(
        # General arguments
        dataset = args.dataset,
        features = int(args.features),
        num_vectors = int(args.num_vectors),
        k_search = int(args.k_search),
        k_result = int(args.k_result),
        skip_init = args.skip_init,
        skip_kmeans = args.skip_global_kmeans,
        kmeans_version = args.kmeans_version,
        implementation = args.impl,
        
        # Custom algorithm arguments
        replication = int(int(args.replication_percentage) * int(args.num_index) / 100) or 1,
        num_index = int(args.num_index),
        num_centroids_search = int(args.num_centroids_search),
        k = int(args.k),
        n_probe = int(args.n_probe),
        query_batch_size = int(args.query_batch_size),

        # Storage
        storage_bucket = args.storage_bucket,

        # Runtime
        index_mem = int(args.indexing_memory),
        search_map_mem = int(args.search_map_memory),
        search_reduce_mem = int(args.search_reduce_memory)
    )
    
    ## Test using our custom indexes
    # Indexing -> Filename and num_workers
    total_times = sv_vectordb.indexing(f'vectors_{args.dataset}.csv', 128)

    query_vectors = []
    if not args.skip_query:
        with open(f'queries_{args.dataset}.csv', mode ='r')as file:
            csvFile = csv.reader(file)
            
            for lines in csvFile:
                vector = lines[0].split(" ")
                vector = [float(value) for value in vector if value != '']
                query_vectors.append(vector)
    
        mult_query = list(divide_chunks(query_vectors, 1000))
        
        storage = Storage()
        res = storage.get_object(bucket=sv_vectordb.params.storage_bucket, key=f"true_neighbours_{args.dataset}.csv").decode("UTF-8")
        csv_buffer = StringIO(res)
        csv_reader = csv.reader(csv_buffer)

        true = []
        for row in csv_reader:
            res_ids = [int(value) for value in row if value != '']
            true.append(res_ids)
    
    shuffle_centroids_times = []
    map_iterdata_times = []
    create_map_data = []
    map_queries_times = []
    map_invocation_times = []
    map_execution_times = []
    create_reduce_times = []
    reduce_iterdata_times = []
    reduce_invocation_times = []
    reduce_queries_times = []
    reduce_execution_times = []
    divide_reduce_times = []
    total_querying_times = []
    recalls = []
    i = 0

    if not args.skip_query:

        for query_vectors in mult_query:
            
            if i == 1:
                break

            smart_neighbours, querying_times = sv_vectordb.search(i, np.array(query_vectors))
                    
            shuffle_centroids_times.append(querying_times[f'{i}_shuffle_{args.impl}'])
            map_iterdata_times.append(querying_times[f'{i}_map_iterdata_{args.impl}'])
            create_map_data.append(querying_times[f'{i}_create_map_data{args.impl}'])
            map_queries_times.append(querying_times[f'{i}_map_{args.impl}'])
            map_execution_times.append(querying_times[f'{i}_map_execution_{args.impl}'])
            create_reduce_times.append(querying_times[f'{i}_create_reduce_data_{args.impl}'])
            reduce_iterdata_times.append(querying_times[f'{i}_reduce_iterdata_{args.impl}'])
            reduce_queries_times.append(querying_times[f'{i}_reduce_{args.impl}'])
            reduce_execution_times.append(querying_times[f'{i}_reduce_execution_{args.impl}'])
            divide_reduce_times.append(querying_times[f'{i}_divide_reduce_{args.impl}'])
            total_querying_times.append(querying_times[f'{i}_total_querying_{args.impl}'])

            map_invocation_times.append(querying_times[f'{i}_map_invocation_{args.impl}'])
            reduce_invocation_times.append(querying_times[f'{i}_reduce_invocation_{args.impl}'])
            
            ## Get our ground truth  
            recalls.append(calculate_mult_recall(true, smart_neighbours))
            
            i += 1
    
    
    total_times['shuffle_centroids_times'] = shuffle_centroids_times
    total_times['map_iterdata_times'] = map_iterdata_times
    total_times['create_map_data'] = create_map_data
    total_times['map_queries_times'] = map_queries_times
    total_times['map_execution'] = map_execution_times
    total_times['map_invoke'] = map_invocation_times
    total_times['create_reduce_data'] = create_reduce_times
    total_times['reduce_iterdata_times'] = reduce_iterdata_times
    total_times['reduce_queries_times'] = reduce_queries_times
    total_times['reduce_execution_times'] = reduce_execution_times
    total_times['reduce_invoke'] = reduce_invocation_times
    total_times['divide_reduce_times'] = divide_reduce_times
    total_times['total_querying_times'] = total_querying_times
    total_times['recalls'] = recalls
    
    if args.impl == "centroids" and not sv_vectordb.params.skip_init:
        total_times[f'distribute_vectors_{args.impl}_mean'] = calculate_mean(total_times[f'distribute_vectors_{args.impl}'])
    
    if not sv_vectordb.params.skip_init:
        total_times[f'generate_index_{args.impl}_mean'] = calculate_mean(total_times[f'generate_index_{args.impl}'])
    if not args.skip_query:
        #total_times['map_queries_times_mean'] = calculate_mean_mult(total_times['map_queries_times'])
        total_times['reduce_queries_times_mean'] = calculate_mean_mult(total_times['reduce_queries_times'])
        total_times['total_querying_times_mean'] = calculate_mean(total_times['total_querying_times'])
        total_times['recalls_mean'] = calculate_mean_mult(total_times['recalls'])
        total_times['number_queries'] = len(query_vectors)
    
    params = {
        'dataset': str(args.dataset),
        'features' : int(args.features),
        'k_search' : int(args.k_search),
        'num_index' : int(args.num_index),
        'replication_percentage': int(args.replication_percentage),
        'num_centroids_search' : int(args.num_centroids_search),
        'k' : int(args.k),
        'kmeans_version': args.kmeans_version,
        'n_probe' : int(args.n_probe),
        'query_batch_size': int(args.query_batch_size),
        'implementation': args.impl,
        'indexing_memory': int(args.indexing_memory),
        'search_map_memory': int(args.search_map_memory),
        'search_reduce_memory': int(args.search_reduce_memory)
    }
    
    total_times['params'] = params
    
    print(total_times)
    
    with open(f'results_{args.dataset}_{args.impl}_{args.num_index}_{int(sv_vectordb.params.num_index/sv_vectordb.params.query_batch_size)}_{int(time.time())}.json', "w") as f:
        json.dump(total_times, f)