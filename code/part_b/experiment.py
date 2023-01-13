import knn_modified
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from utils import *


# Original knn algorithms
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
    nbrs = KNNImputer(n_neighbors=k)
    transpose = np.transpose(matrix)

    mat_tp = nbrs.fit_transform(transpose)
    mat = np.transpose(mat_tp)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    val_acc_e = []
    val_acc2_e = []
    val_acc_h = []
    val_acc2_h = []
    k_values = [10, 20, 30, 40, 50, 60]

    for k in k_values:
        print("User-Based Collaborative Filtering:")
        print("k = " + str(k))
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_h = knn_modified.knn_impute_by_user(sparse_matrix, val_data, k)
        val_acc_e.append(acc)
        val_acc_h.append(acc_h)

        print("Item-Based Collaborative Filtering:")
        print("k = " + str(k))
        acc2 = knn_impute_by_item(sparse_matrix, val_data, k)
        acc2_h = knn_modified.knn_impute_by_item(sparse_matrix, val_data, k)
        val_acc2_e.append(acc2)
        val_acc2_h.append(acc2_h)

    fig1, ax1 = plt.subplots()
    ax1.plot(k_values, val_acc_e, label="Euclidean")
    ax1.plot(k_values, val_acc_h, label="Hamming")
    plt.title("User-Based Collaborative Filtering")
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.legend()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(k_values, val_acc2_e, label="Euclidean")
    ax2.plot(k_values, val_acc2_h, label="Hamming")
    plt.title("Item-Based Collaborative Filtering")
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
