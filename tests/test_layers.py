import os
import sys
print(os.getcwd())
print(sys.path)
sys.path.append("C:\\Users\\super\\Documents\\4.Projects\\neural_network\\network_module")
print(sys.path)
from initializer import NormalInitializer
from layers import Embedding


def test_embedding():
    vocab_size = 10
    embed_dim = 5
    embedding = Embedding(vocab_size, embed_dim)

    # test initialize_params
    embedding.initialize_params(NormalInitializer())
    assert embedding.embedding.value.shape == (vocab_size, embed_dim)

    print(embedding.embedding.value)

    # test forward
    X = [[1, 2, 3], [4, 5, 6]]
    X_embed = embedding.forward(X)
    assert X_embed.shape == (2, 3, embed_dim)

    # test backward
    dX = embedding.backward(X_embed)
    assert dX.shape == (2, 3, vocab_size)