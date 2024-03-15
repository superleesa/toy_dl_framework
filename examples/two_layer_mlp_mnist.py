from preprocessing import get_normalized_data
from layers import Softmax
from model import Model
from layers import Linear, ReLU
from loss import CrossEntropy
from optimizer import SGD
from metric import Accuracy


X_train, y_train, X_test, y_test = get_normalized_data()

layers = [
    Linear(X_train.shape[1], 50),
    ReLU(),
    Linear(50, 10),
    Softmax(),
]

cross_entropy = CrossEntropy()
acc_metric = Accuracy()

mlp = Model(layers)
sgd = SGD(mlp.get_trainable_params(), 0.1)

mlp.fit(X_train, y_train, cross_entropy, sgd, epochs=10)
accuracy = mlp.evaluate(X_test, y_test, acc_metric)
print(accuracy)  # 0.969


