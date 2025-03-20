import faiss
import numpy as np
import csv
import io
import math

class CentroidMaster():
    def __init__(self, centroids, dimensions):
        """Initialise the CentroidMaster with a list of centroids and a Flat index to search them with"""         
        index = faiss.IndexFlatL2(dimensions)
        index.add(np.array(centroids))
        self.centroids = np.array(centroids)
        self.index = index
    
    
    def get_centroid_ids(self, vector, k=1):
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
        vector_count = 0
        if replication == 1:
            for id, vector, label in zip(ids, vectors, labels):
                vector = ' '.join(map(str, vector))
                writers[label].writerow([id, vector])
                vector_count += 1
        else:
            for id, vector, label in zip(ids, vectors, labels):
                max_distance = 2 * np.linalg.norm(vector - self.centroids[label])
                distances, centroid_ids = self.get_centroid_ids(vector, replication)
                vector = ' '.join(map(str, vector))
                writers[label].writerow([id, vector])
                for centroid, distance in zip(centroid_ids[0], distances[0]):
                    if math.sqrt(distance) > max_distance:
                        break
                    if centroid != label:
                        writers[centroid].writerow([id, vector])
                        vector_count += 1

        return buffers, vector_count