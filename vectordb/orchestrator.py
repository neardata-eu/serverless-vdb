from lithops import FunctionExecutor, Storage

from vectordb.config import SvlessVectorDBParams
from .centroids import CentroidMaster
import numpy as np
import time
from .querying import get_mult_neighours, reduce_mult_neighbours

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import orjson

class Orchestrator():
    
    def __init__(self, config: SvlessVectorDBParams, window_time=60):
        self.window_time = window_time
        self.function_executor = FunctionExecutor()
        self.config = config
        self.pool = ThreadPoolExecutor(max_workers=20)  
        
    def shuffle_queries(self, keys, n):
        
        start = time.time()
        if self.config.implementation == "centroids":
            res = self.function_executor.storage.get_object(bucket=self.config.storage_bucket, key=f'indexes/{self.config.dataset}/{self.config.implementation}/{self.config.num_index}/{self.config.centroids_key}')
            centroids = np.array(orjson.loads(res))
            master = CentroidMaster(centroids, len(centroids[0]))
        
        dict = defaultdict(list)
        
        # Modify
        query_id = 0
        
        # ObjStorage -> Pravega or Kinesis
        for key in keys:
            index_ids = master.get_centroid_ids(key, n)[0] if self.config.implementation == "centroids" else list(range(self.config.num_index))
            for id in index_ids:  
                dict[id].append ([query_id, key.tolist()])                                 
            query_id += 1
                
        end = time.time()                
        return dict, end - start
        
        
    def create_map_iterdata(self, payload, batch_size):
        
        map_keys = []
        storage = Storage()
        start = time.time()
        query_info = {}
        local_counter = 0
        filename_counter = 0

        for key, items in payload.items():
            query_info[f'indexes/{self.config.dataset}/{self.config.implementation}/{self.config.num_index}/centroid_{key}.ann'] = items
            local_counter += 1
            if local_counter == batch_size:
                filename = f'queries/batch_{filename_counter}.json'
                self.pool.submit(storage.put_object, self.config.storage_bucket, filename, orjson.dumps(query_info))
                map_keys.append(filename)

                filename_counter += 1
                local_counter = 0
                query_info = {}

        if query_info:
            filename = f'queries/batch_{filename_counter}.json'
            self.pool.submit(storage.put_object, self.config.storage_bucket, filename, orjson.dumps(query_info))
            map_keys.append(filename)
        
        self.pool.shutdown(wait=True)
        end = time.time()
        return map_keys, end - start
    
    
    def create_reduce_iterdata(self, payload, k, num_queries):

        start = time.time()
        storage = Storage()
        reduce_keys = []

        reduce_iterdata = defaultdict(list)
        for query_dict in payload:
            for key, value in query_dict.items():
                reduce_iterdata[key] = reduce_iterdata[key] + value
        sorted_reduce_iterdata = dict(sorted(reduce_iterdata.items()))       
        
        queries = []
        i = 0
        j = 0
        for key, value in sorted_reduce_iterdata.items():
            queries.append([key, value])
            
            i += 1
            
            if i == num_queries:
                key = f'reduce/res_{j}.json'
                storage.put_object(bucket=self.config.storage_bucket, key=key, body=orjson.dumps({"queries": queries, "k": k}))
                reduce_keys.append(key)
                j += 1
                queries = []
                i = 0
                
        if len(queries) > 0:     
            key = f'reduce/res_{j}.json'
            storage.put_object(bucket=self.config.storage_bucket, key=key, body=orjson.dumps({"queries": queries, "k": k}))
            reduce_keys.append(key)
        
        end = time.time()
        return reduce_keys, end - start
    
    
    def divide_map_results(self, futures_res):
        
        results = []
        times = []
        
        for res in futures_res:
            results.append(res[0])
            times.append(res[1])
            
        return results, times
    
    def divide_reduce_results(self, futures_res):
        
        results = []
        times = []
        
        for res in futures_res:
            for q_res in res[0]:
                results.append(q_res)
                
            times.append(res[1])
            
        return results, times
        
    
    def search(self, id_query, queries, n, k_search, k_result):
                
        start = time.time()
        
        # Get centroids
        centroids, shuffle_times = self.shuffle_queries(queries, n)
        
        # Map
        map_keys, map_iterdata_times = self.create_map_iterdata(centroids, self.config.query_batch_size)

        if self.function_executor.config["lithops"]["backend"] == "k8s":
            self.function_executor.config["k8s"]["runtime_cpu"] = self.config.search_map_cpus
            self.function_executor.config["k8s"]["runtime_memory"] = self.config.search_map_mem
            
        index_to_compute = [(x, k_search, self.config) for x in map_keys]
        
        self.function_executor.map(get_mult_neighours, index_to_compute, runtime_memory=self.config.search_map_mem)
        map_futures_res = self.function_executor.get_result()
                                             
        map_res, map_times = self.divide_map_results(map_futures_res)
         
        # Reduce
        reduce_iterdata, reduce_iterdata_times = self.create_reduce_iterdata(map_res, k_result, 1000)
        
        if self.function_executor.config["lithops"]["backend"] == "k8s":
            self.function_executor.config["k8s"]["runtime_cpu"] = self.config.search_reduce_cpus
            self.function_executor.config["k8s"]["runtime_memory"] = self.config.search_reduce_mem
        
        reduce_iterdata = [(x, self.config) for x in reduce_iterdata]
        self.function_executor.map(reduce_mult_neighbours, reduce_iterdata, runtime_memory=self.config.search_reduce_mem)
        reduce_futures_res = self.function_executor.get_result()
        
        reduce_res, reduce_times = self.divide_reduce_results(reduce_futures_res)
        
        end = time.time()
        
        timers = {}
    
        timers[f'{id_query}_shuffle_{self.config.implementation}'] = shuffle_times
        timers[f'{id_query}_map_iterdata_{self.config.implementation}'] = map_iterdata_times
        timers[f'{id_query}_map_{self.config.implementation}'] = map_times
        timers[f'{id_query}_reduce_iterdata_{self.config.implementation}'] = reduce_iterdata_times
        timers[f'{id_query}_reduce_{self.config.implementation}'] = reduce_times
        timers[f'{id_query}_total_querying_{self.config.implementation}'] = end - start
    
        return reduce_res, timers