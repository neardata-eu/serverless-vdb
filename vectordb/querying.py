import faiss
import numpy as np
import orjson
from lithops import Storage
import time
from collections import defaultdict

def get_mult_neighours(queries_key, k, storage: Storage, config):
    """Get the nearest neighbours from an index"""
    start = time.time()
    faiss.omp_set_num_threads(6)
        
    res_queries = defaultdict(list)
    
    if config.implementation == "centroids":
        start_d_queries = time.time()
        queries_json = storage.get_object(bucket=config.storage_bucket, key=queries_key).decode("UTF-8")
        queries_json = orjson.loads(queries_json)
        end_d_queries = time.time()
    
    if config.implementation == "blocks":
        start_d_queries = time.time()
        queries = storage.get_object(bucket=config.storage_bucket, key=queries_key[0])
        queries = orjson.loads(queries)
        queries_json = {f'indexes/{config.dataset}/{config.implementation}/{config.num_index}/centroid_{key}.ann': queries for key in queries_key[1]}
        end_d_queries = time.time()
    
    all_index = []
    all_index_memory = []
    all_a_queries = []
    
    for key, queries in queries_json.items():
        start_index = time.time()
        storage.download_file(config.storage_bucket, key, '/tmp/index.ann')
        end_index = time.time()
        
        all_index.append(end_index - start_index)
        
        res = {}
        res[key] = []
        
        start_index_memory = time.time()
        index = faiss.read_index('/tmp/index.ann')
        end_index_memory = time.time()
        
        all_index_memory.append(end_index_memory - start_index_memory)
        
        start_a_queries = time.time()

        if config.implementation == "centroids":
            query_vectors = [query_data[1] for query_data in queries]
            d, i = index.search(np.array(query_vectors), k)
            for x in range(len(queries)):
                res_queries[queries[x][0]].append([d[x].tolist(), i[x].tolist()])

        if config.implementation == "blocks":
            d, i = index.search(np.array(queries), k)
            for x in range(len(queries)):
                res_queries[x].append([d[x].tolist(), i[x].tolist()])
        
        
        end_a_queries = time.time()
        
        all_a_queries.append(end_a_queries - start_a_queries)
     
    start_reduce_queries = time.time()
    final_results = {}

    filtered_vectors = set()
    for key, res in res_queries.items():
        concat_res = []
        
        for dists, ids in res:
            for dist, id in zip(dists, ids):
                if id not in filtered_vectors:
                    concat_res.append([id, dist])
                    filtered_vectors.add(id)
        best_vectors = sorted(concat_res, key=lambda x: x[1], reverse=False)[:k]
    
        final_results[key] = best_vectors
        
    end_reduce_queries = time.time()

    end = time.time()
    return final_results, [end - start, end_d_queries - start_d_queries, all_index, all_index_memory, all_a_queries, end_reduce_queries - start_reduce_queries]

def reduce_mult_neighbours(reduce_key, storage: Storage, config):
    
    start = time.time()

    res_json = storage.get_object(bucket=config.storage_bucket, key=reduce_key).decode("UTF-8")
    res_json = orjson.loads(res_json)
    
    results = res_json["queries"]
    k = res_json["k"]
    
    final_results = []
    filtered_vectors = set()

    for res in results:
        concat_res = []
        for id, dist in res[1]:
            if id not in filtered_vectors:
                concat_res.append((id, dist))
                filtered_vectors.add(id)
        best_vectors = sorted(concat_res, key=lambda x: x[1], reverse=False)[:k]
        ids = [id for id, _ in best_vectors]
        final_results.append(ids)
    
    end = time.time()
    
    return final_results, end - start