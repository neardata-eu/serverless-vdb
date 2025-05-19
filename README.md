# Serverless Vector Database

## Requirements

### 1. Install Docker

### 2. Install Lithops, FAISS and other required Python libraries
```
pip install  lithops faiss-cpu numpy pandas boto3
```

### 3. Create .lithops_config file
Example with lambda backend:
```
lithops:
 backend: aws_lambda
 storage: aws_s3
 log_level: DEBUG

aws:
    region: YOUR-REGION
    access_key_id: YOUR-ACCESS-KEY-ID
    secret_access_key: YOUR-SECRET-ACCESS-KEY
    session_token: YOUR-SESSION-TOKEN

aws_lambda:
    runtime: YOUR-CUSTOM-LAMBDA-RUNTIME
    execution_role: YOUR-LAMBDA-EXECUTION-ROLE

aws_s3:
    bucket_name: YOUR-S3-BUCKET-NAME
```

Example with k8s backend (example uses minio object storage, but any other service supported by Lithops can be used):
```
lithops:
 backend: k8s
 storage: minio
 log_level: DEBUG
 execution_timeout: 3600

k8s:
 docker_server: docker.io
 docker_user: YOUR-DOCKERHUB-USER
 docker_password: YOUR-DOCKERHUB-PASSWORD
 runtime: YOUR-DOCKERHUB-RUNTIME-NAME
 runtime_memory: 10240
 runtime_cpu: 4
 worker_processes: 1
 runtime_timeout: 3600
 master_timeout: 3600
 namespace: YOUR-K8S-DESIRED-NAMESPACE

minio:
 storage_bucket: YOUR-STORAGE-BUCKET-NAME
 endpoint: http://MINIO-ENDPOINT-IP:9000
 access_key_id: MINIO-USERNAME
 secret_access_key: MINIO-PASSWORD
```
### 4. Dockerhub cli login

### 5. Create runtime for map functions
For lambda backend:
```
lithops runtime build -f Dockerfile.lambda -b aws_lambda YOUR-CONTAINER-RUNTIME-NAME
```

For k8s backend:
```
lithops runtime build -f Dockerfile.k8s -b k8s YOUR-DOCKERHUB-USERNAME/YOUR-CONTAINER-RUNTIME-NAME:TAG
```

### 6. Join parted dataset csv file
```
cat dataseta* > vectors_deep_100k.csv
```

### 7. Upload the vectors csv file and true neighbors csv file to your desired object storage bucket

## Run

### 1. Blocks implementation

```
python3 example.py --features 96 --k_search 10 --k_result 10 --dataset deep_100k --num_index N_PARTITIONS --k 512 --n_probe 32 --query_batch_size N_PARTITIONS/4 --storage_bucket YOUR-STORAGE-BUCKET-NAME --indexing_memory 10240 --search_map_memory 8192 
```

### 2. Centroids implementation

```
python3 example.py --features 96 --k_search 10 --k_result 10 --impl centroids --dataset deep_100k --num_index N_PARTITIONS --num_centroids_search N_SEARCH --k 512 --n_probe 32 --query_batch_size N_PARTITIONS/4 --storage_bucket YOUR-STORAGE-BUCKET-NAME --indexing_memory 10240 --search_map_memory 8192 --num_vectors 100000
```

As explained in configuration section down below, query_batch_size parameter will specify the amount of query map functions to be thrown. N_PARTITIONS/4 will throw 4 functions, while N_PARTITIONS/8 8 functions.

## Configuration

| Parameter              | Description                              | Optional | Default |
| ---------------------- | ---------------------------------------- | -------- | ------- |
| impl                   | Implementation of the serverless vectorDB to be used. | No      | "blocks" or "centroids" |
| num_index              | Number of centroids/blocks into which the vectors are divided/grouped.  | No       |    4     |
| num_centroids_search   | Number of centroids to query.             | No (Centroids)      | 4 |
| features               | Number of vectors features/dimensions.    | No       |         |
| num_vectors               | Number of vectors in the dataset (Only needed on centroids implementation).    | Yes       |   -1      |
| k_search               | Number of neighbours to be searched.      | No       |    10     |
| k_result               | Number of neighbours to be returned.      | No       |    10     |
| storage_bucket         | Storage bucket name.               | No      |  |
| k                      | Number of centroids within the index.     | Yes      | 4096 |
| n_probe                | Number of neighbors searched in the index. | Yes      | 1024 |
| skip_global_kmeans              | Skip global K-means on whole dataset for centroids implementation (only for centroids and labels generation).      | Yes       |    False     |
| kmeans_version              | Implementation of K-means algorithm. It can either be faiss K-means (unbalanced) or a contrained, balanced K-means      | Yes       |    unbalanced     |
| replication_threshold              | Percentage of closest centroid distance to vector for indexing replication when assigning vectors to their corresponding centroid on centroids implementation.      | Yes       |    False     |
| skip_init              | Skip vector database initialization.      | Yes       |    False     |
| skip_query              | Skip querying the vector database.      | Yes       |    False     |
| dataset              | Name of the dataset to be used.      | No       |    glove     |
| query_batch_size              | Number of indexes to be queried per function. Combined with the num_index, it determines the amount of functions to be raised on querying      | Yes       |    16     |
| indexing_memory              | Amount of memory of the indexing map functions      | Yes       |    8192     |
| search_map_memory              | Amount of memory of the get_neighbours map functions      | Yes       |    9216     |
| search_reduce_memory              | Amount of memory of the reduce_neighbours map functions      | Yes       |    2048     |

