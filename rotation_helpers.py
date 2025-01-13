import numpy as np
import pandas as pd

def translate(embeddings: pd.DataFrame) -> pd.DataFrame:
    np_embeddings = embeddings.to_numpy()
    trans_vec = np_embeddings.sum(axis=0) / np_embeddings.shape[0]
    return embeddings - trans_vec


def norm_vectors(embedding: pd.DataFrame) -> pd.DataFrame:
    return embedding.div(embedding.apply(np.linalg.norm, axis=1), axis=0)