from item_response import sigmoid, update_theta_beta
from utils import *
import numpy as np


def predict(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def bootstrap(data):
    length = len(data["user_id"])
    index = np.random.randint(0, length, length)

    bag = {}
    for key in data.keys():
        bag[key] = np.array(data[key])[index]
    return bag


def irt(data, lr, iterations):
    theta = np.zeros(542)
    beta = np.zeros(1774)

    for _ in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)
    return theta, beta


def average_predict(data, lst):
    predictions = np.zeros(len(data['is_correct']))
    for model in lst:
        predictions += predict(data, model[0], model[1])
    return predictions / len(lst)


def evaluate(predictions, targets):
    preds = np.array([0 if pred < 0.5 else 1 for pred in predictions])
    arr = (preds - np.array(targets) == 0)
    correct = np.count_nonzero(arr)
    return correct / len(targets)


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    total_models = 3
    list = []

    for _ in range(total_models):
        bag = bootstrap(train_data)
        list.append(irt(bag, 0.01, 100))

    train_predictions = average_predict(train_data, list)
    train_accuracy = evaluate(train_predictions, train_data["is_correct"])

    val_predictions = average_predict(val_data, list)
    val_accuracy = evaluate(val_predictions, val_data["is_correct"])

    test_predictions = average_predict(test_data, list)
    test_accuracy = evaluate(test_predictions, test_data["is_correct"])

    print(f'Final Training Accuracy: {train_accuracy}')
    print(f'Final Validation Accuracy: {val_accuracy}')
    print(f'Final Test Accuracy: {test_accuracy}')


if __name__ == "__main__":
    main()
