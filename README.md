# BERT4Rec implementation and demonstration

This is a demonstration of the BERT4Rec model from the [BERT4Rec paper](https://arxiv.org/abs/1904.06690).

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
