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
    
    def get_centroid_ids_with_distance(self, vector, k=1):
        """Get the nearest centroids to a vector"""
        vector = vector.reshape(1, -1)
        distances, indices = self.index.search(vector, k)
        return distances, indices
            
    def generate_csvs(self, ids, vectors, num_centroids, replication, labels):
        """Generate CSV files (in memory) for each centroid and add the vectors to them"""
        writers = []
        buffers = []
        for _ in range(num_centroids):
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            writers.append(csv_writer)
            buffers.append(csv_buffer)
        
        if replication == 1:
            for id, vector, label in zip(ids, vectors, labels):
                vector = ' '.join(map(str, vector))
                writers[label].writerow([id, vector])
        else:
            for id, vector, label in zip(ids, vectors, labels):
                distances, centroid_ids = self.get_centroid_ids_with_distance(vector, replication)
                vector = ' '.join(map(str, vector))
                
                writers[label].writerow([id, vector])
                for centroid, distance in zip(centroid_ids[0], distances[0]):
                    if centroid != label and distance <= 0.95:
                        writers[centroid].writerow([id, vector])
        return buffers