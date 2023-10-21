from model import Model
from layers import Linear, ReLU
from loss import SoftmaxThenCrossEntropy
from initializer import RandomInitializer
from optimizer import SGD

# for toy data
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
data = load_wine()
X = data["data"]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

layers = [
    Linear(X.shape[1], 10),
    ReLU(),
    Linear(10, 3),
    ReLU(),
]

# todo initialize weights randomly but initialize bias with zeros

cross_entropy = SoftmaxThenCrossEntropy()


mlp = Model(layers)
sgd = SGD(mlp.get_trainable_params(), 0.01)

mlp.fit(X_train, y_train, cross_entropy, sgd, epochs=1, initializer=RandomInitializer())
