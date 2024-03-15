import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0].joinpath('src')))
print(str(Path(__file__).parents[0].joinpath('src')))
from layers import Embedding
import numpy as np

def test_embedding():
    vocab_size = 10
    embed_dim = 5
    embedding = Embedding(vocab_size, embed_dim)

    # test initialize_params
    embedding.initialize_params()
    assert embedding.embedding.value.shape == (vocab_size, embed_dim)

    print(embedding.embedding.value)

    # test forward
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X_embed = embedding.forward(X)
    assert X_embed.shape == (2, 3, embed_dim)

    print(X_embed.flatten().shape)
    print(X.flatten().shape)
    # test backward (assume X_embed is gradient of loss with respect to X_embed)
    dX = embedding.backward(X_embed)
    print(dX)
    assert dX.shape == (vocab_size, embed_dim)