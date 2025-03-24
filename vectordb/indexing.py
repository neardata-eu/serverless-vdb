import faiss
import numpy as np
import csv
from k_means_constrained import KMeansConstrained

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
    
    ## Download vectors
    start = time.time()
    
    if params.kmeans_version == "balanced":
        # Use KMeansConstrained to generate centroids of a balanced KMeans cluster
        clf = KMeansConstrained(
            n_clusters=params.num_index,
            size_min=int(len(vectors)/params.num_index*0.9),
            size_max=int(len(vectors)/params.num_index*1.1),
            random_state=0,
            init='k-means++',
            max_iter=300,
            tol=0.0001,
            verbose=False,
            n_jobs=128
        )
        clf.fit(vectors)
        centroids = clf.cluster_centers_
        labels = clf.labels_

    elif params.kmeans_version == "unbalanced":
        faiss.omp_set_num_threads(128)

        index = faiss.Kmeans(vectors.shape[1], params.num_index, niter=20, verbose=True)#, spherical=True)
        index.train(vectors)
        centroids = index.centroids

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(centroids)
        _, labels = index.search(vectors, 1)
        labels = labels.flatten()

    #2025-03-11T13:31:47 - Started clustering
    #2025-03-11T14:38:36 - Finished clustering

    ## Upload centroids
    serialized_data = orjson.dumps(centroids.tolist())

    storage.put_object(bucket=params.storage_bucket, key=f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/{params.centroids_key}', body=serialized_data)
    
    
    serialized_data = orjson.dumps(labels.tolist())
    storage.put_object(bucket=params.storage_bucket, key=f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/{params.labels_key}', body=serialized_data)

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
        
        # Get labels from global K-means
        res = storage.get_object(bucket=params.storage_bucket, key=f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/{params.labels_key}')
        labels = np.array(orjson.loads(res))
        labels = [labels[x] for x in ids]

        # Distribute vectors across the different centroids
        writers, counts = master.generate_csvs(np.array(ids), np.array(vectors), len(centroids), params.replication, labels)
        ## Upload csvs to storage
        for i, buffer in enumerate(writers):
            storage.put_object(bucket=params.storage_bucket, key=f'centroids/{params.dataset}/{params.implementation}/{params.num_index}/{i}/centroid_{id}.csv', body=buffer.getvalue())
            
        end = time.time()
        
        return end - start, counts

def generate_index_centroids(index_ids, params, storage: Storage):
    """Generate an index from a CSV file"""
    faiss.omp_set_num_threads(6)
    ## Download Vectors
    start = time.time()
    
    for index_id in index_ids:
        
        keys = storage.list_keys(bucket=params.storage_bucket, prefix=f'centroids/{params.dataset}/{params.implementation}/{params.num_index}/{index_id}/')
        
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


def get_vectors_with_ids(args):
    
    start, size = args

    df = pd.read_csv(
        '/tmp/vectors.csv',
        header=None,
        skiprows=start,
        nrows=size
    )

    ids = df[0].tolist()
    vectors = [[float(x) for x in s.split() if x] for s in df[1]]
    del df
    return ids, vectors


def initialize_database(filename, params: SvlessVectorDBParams, fexec, num_workers):
    
    init = time.time()
    
    if params.implementation == "centroids" and not params.skip_kmeans:

        logging.info("Uploading file")
        storage = Storage()
        storage.download_file(params.storage_bucket, filename, '/tmp/vectors.csv')
        download = time.time()
        vector_count = params.num_vectors
        chunk_size, remainder = divmod(vector_count, num_workers)
        
        chunks = []
        start = 0
        for i in range(num_workers):
            current_size = chunk_size + (1 if i < remainder else 0)
            chunks.append((start, current_size))
            start += current_size
        prepare_chunks = time.time()

        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(get_vectors_with_ids, chunks)
        get_vectors = time.time()

        ids, vectors = [], []
        for ids_part, vectors_part in results:
            ids.extend(ids_part)
            vectors.extend(vectors_part)
        del results
        combine = time.time()
        
        load_dataset_time = [time.time() - init, download-init, prepare_chunks-download, get_vectors-prepare_chunks, combine-get_vectors]
        
    logging.info("Starting indexing")
    
    ## Distribute dataset across the different centroids
    vectors_key = params.storage_bucket + "/" + filename
    
    if params.implementation == "centroids":
        ## Generate centroids
        if not params.skip_kmeans:
            global_index_time = create_global_index(np.array(vectors), params, fexec.storage)
        futures = fexec.map(distribute_vectors_centroids, vectors_key, extra_args=[params], obj_chunk_number=16, runtime_memory=params.index_mem)
        output = fexec.get_result()
        distribute_vectors_time, indexed_vectors = map(list, zip(*output))
        lambda_invocation_distribute = [f.stats["worker_func_start_tstamp"] - f.stats["host_job_create_tstamp"] for f in futures]
        
    ## Generate an index for each centroid with the vectors assigned to it
    if params.implementation == "centroids":
        all_index = list(range(params.num_index))
        index_to_compute = [all_index[x:x+int(params.num_index/16)] for x in range(0, len(all_index),int(params.num_index/16))]
        futures = fexec.map(generate_index_centroids, index_to_compute, extra_args=[params], runtime_memory=params.index_mem)
        
    elif params.implementation == "blocks":
        
        obj_chunk = 16
        n_blocks_per_function = int(params.num_index / obj_chunk)
        futures = fexec.map(generate_index_blocks, vectors_key, extra_args=[params, n_blocks_per_function], obj_chunk_number=obj_chunk, runtime_memory=params.index_mem)
        
        
    generate_index_time = fexec.get_result()
    lambda_invocation_indexing = [f.stats["worker_func_start_tstamp"] - f.stats["host_job_create_tstamp"] for f in futures]
    
    # Lithops plots
    end = time.time()
    
    timers = {}
    if params.implementation == "centroids":
        timers[f'load_dataset_{params.implementation}'] = load_dataset_time if not params.skip_kmeans else 0
        timers[f'global_index_{params.implementation}'] = global_index_time if not params.skip_kmeans else 0
        timers[f'distribute_vectors_{params.implementation}'] = distribute_vectors_time
        timers[f'distribute_vectors_invocation_{params.implementation}'] = lambda_invocation_distribute
        timers[f'total_generated_vectors'] = sum(indexed_vectors)
        
    
    timers[f'generate_index_{params.implementation}'] = generate_index_time
    timers[f'generate_index_invocation_{params.implementation}'] = lambda_invocation_indexing
    timers[f'total_indexing_{params.implementation}'] = end - init
    
    return timers