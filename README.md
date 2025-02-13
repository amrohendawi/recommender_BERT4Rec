# BERT4Rec implementation and demonstration

This is a demonstration of the BERT4Rec model from the [BERT4Rec paper](https://arxiv.org/abs/1904.06690).


<!--  a sequence of movies as input and a recommendation as output -->
```mermaid
graph TD
    A[User] -->|list: input watch history| B[Black Panther 2017, The Avengers 2012, ...  Spider-Man 3]
    A[User] -->|int: top n recommendations| B1[3]
    B --> |convert movie strings to indices| B2[113, 22, .. 598]
    B2 --> C{BERT4Rec Movies Recommender}
    B1 --> C{BERT4Rec Movies Recommender}
    C -->|list: integer| D1[13, 976, 49]
    D1 --> |convert indices to movie strings| D[Thor: Ragnarok 2017, Deadpool 2 2018, Incredibles 2]
```

### How it works



### Setup (GPU)

Tested on NVIDIA RTX 3070 with CUDA 11.6 and Python 3.9.13

#### 1. Create a virtual environment:

```
python -m venv venv
```

#### 2. Activate the virtual environment:

<details>
  <summary>For Linux</summary>

  ```
  source venv/bin/activate
  ```

</details>

<details>
  <summary>For Powershell</summary>

  ```
  .\venv\Scripts\activate
  ```

</details>

#### 3. Install the dependencies:

```
pip install -r requirements.txt
```

### Training

1. Download the movies dataset from [here](https://grouplens.org/datasets/movielens/25m/).
2. Store the unzipped data in *data* folder.
3. Run the following command to train the model:

```
python3  recommender/training.py --data_csv_path <path-to-the-data>
```

### Docker (CPU)

```bash
docker build . -t recommender
docker run recommender sh -c "python3.8 -m pytest"
```

### References

- [Blog](https://towardsdatascience.com/build-your-own-movie-recommender-system-using-bert4rec-92e4e34938c5)
- [BERT4Rec: Sequential Recommendation with Bidirectional
  Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)
