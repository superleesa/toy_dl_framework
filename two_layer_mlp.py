from basic_layers import Softmax
from model import Model
from layers import Linear, ReLU
from loss import SoftmaxThenCrossEntropy, CrossEntropy
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
    Softmax(),
]

# todo initialize weights randomly but initialize bias with zeros

cross_entropy = CrossEntropy()


mlp = Model(layers)
sgd = SGD(mlp.get_trainable_params(), 0.01)

mlp.fit(X_train, y_train, cross_entropy, sgd, epochs=1, initializer=RandomInitializer(), num_iterations=1000)
y_pred = mlp.predict(X_test)
correct_count = (y_pred.argmax(axis=1) == y_test).sum()
accuracy = correct_count / len(y_pred)
print(accuracy)
