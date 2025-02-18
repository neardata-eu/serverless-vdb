from dataclasses import dataclass

@dataclass
class SvlessVectorDBParams:
    
    # General arguments
    dataset: str = "glove"
    features: int = 64
    k_search: int = 5
    k_result: int = 5
    skip_init: bool = False
    implementation: str = "blocks"
    
    # Custom algorithm arguments
    num_index: int = 4
    num_centroids_search: int = 4
    k: int = 4096
    n_probe: int = 1024
    query_batch_size: int = 16,
    
    # Storage
    storage_bucket: str = None
    centroids_key: str = "centroids.json"
    
    # Runtime
    index_mem: int = 8192
    search_map_cpus: int = 6
    search_map_mem: int = 9216
    search_reduce_cpus: int = 1
    search_reduce_mem: int = 2048
