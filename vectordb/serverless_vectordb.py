from .config import SvlessVectorDBParams
from lithops import FunctionExecutor

from .indexing import initialize_database
from .orchestrator import Orchestrator

class ServerlessVectorDB():
    
    def __init__(self, **parameters):
        self.params: SvlessVectorDBParams = SvlessVectorDBParams(**parameters)
        self.indexing_executor = FunctionExecutor()
        self.orchestrator = Orchestrator(self.params)
        
    def indexing(self, filename, num_workers):
        if not self.params.skip_init:
            return initialize_database(filename, self.params, self.indexing_executor, num_workers)
        return {}
        
    def search(self, id, query_vector):
        return self.orchestrator.search(id, query_vector, self.params.num_centroids_search, self.params.k_search, self.params.k_result)