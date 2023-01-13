import math
import matplotlib.pyplot as plt
from utils import *
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    sum = 0
    num = len(data["user_id"])
    for i in range(num):
        num = math.exp((theta[data["user_id"][i]] - beta[data["question_id"][i]])
                       * data["is_correct"][i])
        den = 1 + math.exp(theta[data["user_id"][i]] - beta[data["question_id"][i]])
        sum = sum + math.log((num / den))

    log_lklihood = sum
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num = len(data["user_id"])
    for i in range(num):
        x = theta[data["user_id"][i]] - beta[data["question_id"][i]]
        sigmoid = math.exp(x) / (1 + math.exp(x))

        update = sigmoid - data["is_correct"][i]
        theta[data["user_id"][i]] = theta[data["user_id"][i]] - (lr * update)

        update = data["is_correct"][i] - sigmoid
        beta[data["question_id"][i]] = beta[data["question_id"][i]] - (lr * update)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    train_log_lik = []
    val_log_lik = []
    val_acc_lst = []

    for _ in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld2 = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_log_lik.append(-neg_lld)
        val_log_lik.append(-neg_lld2)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_log_lik, val_log_lik, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Uncomment to tune hyperparameters
    # lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    # num_iterations = [50, 100, 250, 500, 1000]
    #
    # max_acc = 0
    # max_acc_lr = lrs[0]
    # max_acc_itr = num_iterations[0]
    # for lr in lrs:
    #     for i in num_iterations:
    #         _, _, val_acc_lst = irt(train_data, val_data, lr, i)
    #         if val_acc_lst[-1] > max_acc:
    #             max_acc = val_acc_lst[-1]
    #             max_acc_lr = lr
    #             max_acc_itr = i
    #
    # print(max_acc_lr, max_acc_itr)

    # Chosen hyperparameters are
    # learning rate: 0.005
    # number of iterations: 500

    _, beta, train_log_lik, val_log_lik, val_acc_lst = irt(train_data, val_data, 0.005, 500)

    fig1, ax1 = plt.subplots()
    ax1.plot([x for x in range(500)], train_log_lik, label="Training log-likelihood")
    ax1.plot([x for x in range(500)], val_log_lik, label="Validation log-likelihood")
    ax1.legend()
    plt.title("Training Curve")
    plt.ylabel('Log-likelihood')
    plt.xlabel('Number of iterations')
    plt.show()

    _, _, _, _, test_acc_lst = irt(train_data, test_data, 0.005, 500)

    print("Final validation accuracy:")
    print(val_acc_lst[-1])

    print("Final test accuracy:")
    print(test_acc_lst[-1])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    #####################################################################
    fig1, ax1 = plt.subplots()
    theta_val = [x for x in range(-5, 5)]
    ax1.plot(theta_val, sigmoid(theta_val - beta[1]), color='red', label='j1')
    ax1.plot(theta_val, sigmoid(theta_val - beta[17]), color='orange', label='j2')
    ax1.plot(theta_val, sigmoid(theta_val - beta[20]), color='green', label='j3')
    plt.xlabel('Theta')
    plt.ylabel('Probability of Correct Response')
    plt.legend()
    plt.title('Probability of Correctly Answering Questions (j1, j2, j3)')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
