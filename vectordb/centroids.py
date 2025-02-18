import faiss
import numpy as np
import csv
import io

class CentroidMaster():
    def __init__(self, centroids, dimensions):
        """Initialise the CentroidMaster with a list of centroids and a Flat index to search them with"""         
        index = faiss.IndexFlatL2(dimensions)
        index.add(np.array(centroids))
        self.index = index
    
    
    def get_centroid_ids(self, vector, k=1):
        """Get the nearest centroids to a vector"""
        vector = vector.reshape(1, -1)
        _, indices = self.index.search(vector, k)
        return indices
            
            
    def generate_csvs(self, ids, vectors, num_centroids):
        """Generate CSV files (in memory) for each centroid and add the vectors to them"""
        writers = []
        buffers = []
        for _ in range(num_centroids):
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            writers.append(csv_writer)
            buffers.append(csv_buffer)
            
        for id, vector in zip(ids, vectors):
            centroid_id = self.get_centroid_ids(vector)[0][0]
            vector = ' '.join(map(str, vector))
            writers[centroid_id].writerow([id, vector])
            
        return buffers