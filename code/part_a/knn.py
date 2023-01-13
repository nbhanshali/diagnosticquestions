import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    transpose = np.transpose(matrix)

    mat_tp = nbrs.fit_transform(transpose)
    mat = np.transpose(mat_tp)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    val_acc = []
    val_acc2 = []
    k_values = [1, 6, 11, 16, 21, 26]

    for k in k_values:
        print("User-Based Collaborative Filtering:")
        print("k = " + str(k))
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        val_acc.append(acc)

        print("Item-Based Collaborative Filtering:")
        print("k = " + str(k))
        acc2 = knn_impute_by_item(sparse_matrix, val_data, k)
        val_acc2.append(acc2)

    fig1, ax1 = plt.subplots()
    ax1.plot(k_values, val_acc)
    plt.title("User-Based Collaborative Filtering")
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(k_values, val_acc2)
    plt.title("Item-Based Collaborative Filtering")
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.show()

    argmax_val_acc = val_acc.index(max(val_acc))
    k_star = k_values[argmax_val_acc]

    argmax_val_acc2 = val_acc2.index(max(val_acc2))
    k_star2 = k_values[argmax_val_acc2]

    print("User-Based Collaborative Filtering:")
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print("k* = " + str(k_star))
    print("Test Accuracy for k*:")
    print(test_acc)

    print("Item-Based Collaborative Filtering:")
    test_acc2 = knn_impute_by_item(sparse_matrix, test_data, k_star2)
    print("k* = " + str(k_star2))
    print("Test Accuracy for k*:")
    print(test_acc2)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
