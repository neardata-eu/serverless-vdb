import faiss
import numpy as np
import csv

from vectordb.config import SvlessVectorDBParams
from .centroids import CentroidMaster
import orjson
from lithops import Storage
from io import StringIO
import time
import pandas as pd
import multiprocessing
import logging

def create_global_index(vectors, params, storage: Storage):
    """Distribute vectors into different centroids by using a Master index"""
    
    faiss.omp_set_num_threads(128)
    ## Download vectors
    start = time.time()
    
    ## Use faiss to generate centroids
    #index = faiss.index_factory(params.features, f"IVF{params.num_index},Flat")
    index = faiss.Kmeans(vectors.shape[1], params.num_index, niter=20, verbose=True, spherical=True)
    index.train(vectors)
    #centroids = index.quantizer.reconstruct_n(0, params.num_index)
    centroids = index.centroids
    
    ## Upload centroids
    serialized_data = orjson.dumps(centroids.tolist())

    storage.put_object(bucket=params.storage_bucket, key=f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/{params.centroids_key}', body=serialized_data)
    end = time.time()
        
    return end-start


def distribute_vectors_centroids(id, obj, params, storage: Storage):
    if params.implementation == "centroids":
        faiss.omp_set_num_threads(6)

        # Get the chunk of vectors assigned to this function
        start = time.time()

        csv_data = obj.data_stream.read().decode('utf-8')
        csv_buffer = StringIO(csv_data)
        csv_reader = csv.reader(csv_buffer)
        vectors = []
        ids = []
        for row in csv_reader:
            vector = row[1].split(" ")
            vector = [float(value) for value in vector if value != '']
            vectors.append(vector)
            ids.append(int(row[0]))
        
        # Get centroids to generate the master
        res = storage.get_object(bucket=params.storage_bucket, key=f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/{params.centroids_key}')
        centroids = np.array(orjson.loads(res))
        master = CentroidMaster(centroids, params.features)
        
        # Distribute vectors across the different centroids
        writers = master.generate_csvs(np.array(ids), np.array(vectors), len(centroids))

        ## Upload csvs to storage
        for i, buffer in enumerate(writers):
            storage.put_object(bucket=params.storage_bucket, key=f'centroids/{i}/centroid_{id}.csv', body=buffer.getvalue())
            
        end = time.time()
        
        return end - start

def generate_index_centroids(index_ids, params, storage: Storage):
    """Generate an index from a CSV file"""
    faiss.omp_set_num_threads(6)
    ## Download Vectors
    start = time.time()
    
    for index_id in index_ids:
        
        keys = storage.list_keys(bucket=params.storage_bucket, prefix=f'centroids/{index_id}/')
        
        ids = []
        vectors = []
        for key in keys:
            res = storage.get_object(bucket=params.storage_bucket, key=key).decode('utf-8')
            csv_file = StringIO(res)
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                vector = row[1].split(" ")
                vector = [float(value) for value in vector if value != '']
                vectors.append(vector)
                ids.append(int(row[0]))
        
        if len(vectors) > 128:
            ## Generate Flat Index
            index = faiss.index_factory(params.features, f'IVF{params.k},Flat')
            index.train(np.array(vectors)) 
            index.nprobe = params.n_probe
            index.add_with_ids(np.array(vectors), np.array(ids))
            
        else:
            index = faiss.IndexFlatL2(params.features)
            index = faiss.IndexIDMap(index)
            index.add_with_ids(np.array(vectors), np.array(ids))
        
        ## Store index to disk
        faiss.write_index(index, f'/tmp/{index_id}.ann')
        
        ## Upload index to storage
        storage.upload_file(f'/tmp/{index_id}.ann', params.storage_bucket, f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/centroid_{index_id}.ann')
    
    end = time.time()
    return end - start

def generate_index_blocks(id, obj, params, n_blocks, storage: Storage):
    """Generate an index from a CSV file"""
    faiss.omp_set_num_threads(6)
    ## Download Vectors
    start = time.time()
        
    csv_data = obj.data_stream.read().decode('utf-8')
    csv_buffer = StringIO(csv_data)
    csv_reader = csv.reader(csv_buffer)
    
    csv_reader = list(csv_reader)
    
    quotient, remainder = divmod(len(csv_reader), n_blocks)
    lower_elements = [quotient for i in range(n_blocks - remainder)]
    higher_elements = [quotient + 1 for j in range(remainder)]
    n_vecs_per_block = lower_elements + higher_elements
    
    i = 0
    key_id = id * n_blocks
    ids = []
    vectors = []

    for row in csv_reader:
        vector = row[1].split(" ")
        vector = [float(value) for value in vector if value != '']
        vectors.append(vector)
        ids.append(int(row[0]))

        if len(vectors) == n_vecs_per_block[i]:
            index = faiss.index_factory(params.features, f'IVF{params.k},Flat')
            index.train(np.array(vectors)) 
            index.nprobe = params.n_probe
            index.add_with_ids(np.array(vectors), np.array(ids))
                
            ## Store index to disk
            faiss.write_index(index, f'/tmp/{key_id}.ann')
            
            ## Upload index to storage
            storage.upload_file(f'/tmp/{key_id}.ann', params.storage_bucket, f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/centroid_{key_id}.ann')

            ids = []
            vectors = []
                
            key_id += 1
            i += 1
    
    end = time.time()
    return end - start


def get_vectors_with_ids(df):
    
    vectors = df.values.tolist()
        
    new_vectors = []
    ids = []
    for vector in vectors:
        
        ids.append(int(vector[0]))
        vector = vector[1].split(" ")
        vector = [float(value) for value in vector if value != '']
        new_vectors.append(vector)
        
    return ids, new_vectors


def sep_results(results):
    
    ids = []
    vectors = []
    for res in results:
        ids.append(res[0])
        vectors.append(res[1])
        
    return ids, vectors


def list_to_list(data):
    return [item for row in data for item in row]

def initialize_database(filename, params: SvlessVectorDBParams, fexec, num_workers):
    
    start = time.time()
    
    if params.implementation == "centroids":

        logging.info("Uploading file")
        storage = Storage()
        storage.download_file(params.storage_bucket, filename, '/tmp/vectors.csv')
        df = pd.read_csv('/tmp/vectors.csv', header=None)
        
        logging.info("Extracting vectors")
        df = np.array_split(df, num_workers)
        
        pool = multiprocessing.Pool(processes=num_workers)
        res = pool.map(get_vectors_with_ids, df)
        pool.close()
        
        ids, vectors = sep_results(res)
        
        del res
        del df
        
        ids = list_to_list(ids)
        vectors = list_to_list(vectors)
        load_dataset_time = time.time() - start
    
    logging.info("Starting indexing")

    ## Distribute dataset across the different centroids
    vectors_key = params.storage_bucket + "/" + filename
    
    if params.implementation == "centroids":
        ## Generate centroids
        global_index_time = create_global_index(np.array(vectors), params, fexec.storage)
        fexec.map(distribute_vectors_centroids, vectors_key, extra_args=[params], obj_chunk_size=int(512 * pow(2,20)), runtime_memory=params.index_mem)
        distribute_vectors_time = fexec.get_result()
    
    ## Generate an index for each centroid with the vectors assigned to it
    if params.implementation == "centroids":
        all_index = list(range(params.num_index))
        index_to_compute = [all_index[x:x+int(params.num_index/16)] for x in range(0, len(all_index),int(params.num_index/16))]
        fexec.map(generate_index_centroids, index_to_compute, extra_args=[params], runtime_memory=params.index_mem)
        
    elif params.implementation == "blocks":
        
        obj_chunk = 16
        n_blocks_per_function = int(params.num_index / obj_chunk)
        fexec.map(generate_index_blocks, vectors_key, extra_args=[params, n_blocks_per_function], obj_chunk_number=obj_chunk, runtime_memory=params.index_mem)
        
        
    generate_index_time = fexec.get_result()
    
    # Lithops plots
    end = time.time()
    
    timers = {}
    if params.implementation == "centroids":
        timers[f'load_dataset_{params.implementation}'] = load_dataset_time
        timers[f'global_index_{params.implementation}'] = global_index_time
        timers[f'distribute_vectors_{params.implementation}'] = distribute_vectors_time
    
    timers[f'generate_index_{params.implementation}'] = generate_index_time
    timers[f'total_indexing_{params.implementation}'] = end - start
    
    return timers