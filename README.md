# Serverless Vector Database

## Run

### 1. Blocks implementation

```
python3 example.py --impl blocks --num_index 8
```

### 2. Centroids implementation

```
python3 example.py --impl centroids --num_index 8 --num_centroids_search 4
```

## Configuration

| Parameter              | Description                              | Optional | Default |
| ---------------------- | ---------------------------------------- | -------- | ------- |
| impl                   | Implementation of the serverless vectorDB to be used. | No      | "blocks" or "centroids" |
| num_index              | Number of centroids/blocks into which the vectors are divided/grouped.  | No       |    4     |
| num_centroids_search   | Number of centroids to query.             | No (Centroids)      | 4 |
| features               | Number of vectors features/dimensions.    | No       |         |
| k_search               | Number of neighbours to be searched.      | No       |    10     |
| k_result               | Number of neighbours to be returned.      | No       |    10     |
| storage_bucket         | Storage bucket name.               | No      |  |
| k                      | Number of centroids within the index.     | Yes      | 4096 |
| n_probe                | Number of neighbors searched in the index. | Yes      | 1024 |
| skip_init              | Skip vector database initialization.      | Yes       |    False     |
| skip_query              | Skip querying the vector database.      | Yes       |    False     |
| dataset              | Name of the dataset to be used.      | No       |    glove     |
| query_batch_size              | Number of indexes to be queried per function. Combined with the num_index, it determines the amount of functions to be raised on querying      | Yes       |    16     |
| indexing_memory              | Amount of memory of the indexing map functions      | Yes       |    8192     |
| search_map_memory              | Amount of memory of the get_neighbours map functions      | Yes       |    9216     |
| search_reduce_memory              | Amount of memory of the reduce_neighbours map functions      | Yes       |    2048     |


