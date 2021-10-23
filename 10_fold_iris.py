import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# method that will load the data from the URL passed in as parameters
# must specify which of two class labels will be encoded as 0
def load_data(url, to_zero):
    # load the data from the specified url
    data = pd.read_csv(url, header=None)
    # print the intial data
    print(data)

    # encode to_zero class label as 0
    data[4] = np.where(data.iloc[:, -1] == to_zero, 0, 1)
    # print the data again to see the changes
    print(data)

    return data


# method that will be used to create the 10 folds for cross validation
def create_10_folds(data):
    # taking a random sample of the entire dataframe ie shuffling the values randomly
    shuffleDf = data.sample(frac=1)
    # print the shuffle df showing that the new random ordering
    print(shuffleDf)
    # now that the df is such that the ordering of the samples is completely random in relation to the original ordering
    # split the df into 10 subsets that are random due to the manner in which samples were shuffled
    folds = np.array_split(shuffleDf, 10)

    # # make data in folds numpy mx so operations are computationally optimized
    # for i in range(len(folds)):
    #     folds[i] = np.asmatrix(folds[i], dtype='float64')

    return folds


# method that will perform 10 fold cross validation
def ten_fold_cross_validatoin(folds):
    for i in range(len(folds)):
        # print the fold being tested
        print("Performing tests on fold", i+1)
        # obtain the training set from all folds other than that being tested on
        train = np.concatenate((folds[:i] + folds[i + 1:]))
        # visualizing the linear separability of the training data
        visualize_lin_sep_iris(train)
        # converting the train to a np matrix to optimize operations
        train = np.asmatrix(train, dtype='float64')
        # fold i is being tested on
        test = np.asmatrix(folds[i], dtype='float64')


# method that will help visualize that the data is in fact linearly separable
# this is specifically for the iris data set
def visualize_lin_sep_iris(train):
    df = pd.DataFrame(train)
    mask = df[4] == 0
    setosa = df[mask]
    versicolor = df[~mask]
    plt.scatter(np.array(setosa.iloc[:, 0]), np.array(setosa.iloc[:, 2]), marker='o', label='setosa')
    plt.scatter(np.array(versicolor.iloc[:, 0]), np.array(versicolor.iloc[:, 2]), marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend()
    plt.show()

# method that will train a perceptron on the data
def perceptron(data, num_iter):
    # the features are all attributes besides the class label that is ind -1
    features = data[:, :-1]
    # class labels are at index -1
    labels = data[:, -1]

    # beginning training by setting the weights all to 0 and also not the addition of the bias term
    w = np.zeros(shape=(1, features.shape[1] + 1))

    # list that will be used to store the number of misclassified examples at each iteration in training
    misclassified_list = []

    # train for the specified number of epochs
    for epoch in range(num_iter):
        # this will accumulate the number of misclassified examples for this epoch
        misclassified = 0
        # for each epoch go over all training examples
        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            # obtain a prediction for the example
            y = np.dot(w, x.transpose())
            # if y > 0 predict Iris-virginica
            # uf y <= 0 predict Iris-setosa
            target = 1.0 if (y > 0) else 0.0

            # compute delta for updating the weight matrix
            delta = (label.item(0, 0) - target)

            # if the example was misclassified ie delta != 0, update the weights and increment misclassified
            if (delta):
                misclassified += 1
                w += (delta * x)
        # for each iteration append the num misclassified
        misclassified_list.append(misclassified)

    return w, misclassified_list


# method that will display the misclassified over epochs
def misclass_over_its(iterations, misclassified_list):
    epochs = np.arange(1, iterations + 1)
    plt.plot(epochs, misclassified_list)
    plt.xlabel('iterations')
    plt.ylabel('misclassified')
    plt.show()

# method that will be used to obtain the data to test on which is the data after that which
# we already trained on knowing it was linearly separable
def load_data_test(url,index_lin_seperable, to_zero):
    # load the data from the specified url
    data = pd.read_csv(url, header=None)

    # load all of the data after the index that was specified that the data
    # was lin sep up to
    data = data[index_lin_seperable:]

    # print the new data
    print(data)

    # encode to_zero class label as 0
    data[4] = np.where(data.iloc[:, -1] == to_zero, 0, 1)
    # print the data again to see the changes
    print(data)
    # make data a numpy mx so operations are computationally optimized
    data = np.asmatrix(data, dtype='float64')
    return data

# method that will return performance measures for perceptron
# given trained weights and data to test on
def perceptron_test(data, w):
    # the features are all attributes besides the class label that is ind -1
    features = data[:, :-1]
    # class labels are at index -1
    labels = data[:, -1]

    # iterate over the examples and count misclassifications
    correct = 0
    incorrect = 0
    for x, label in zip(features, labels):
        x = np.insert(x, 0, 1)
        # obtain a prediction for the test example
        y = np.dot(w, x.transpose())
        # if y > 0 predict Iris-virginica
        # uf y <= 0 predict Iris-setosa
        target = 1.0 if (y > 0) else 0.0

        if target == label.item(0, 0):
            correct += 1
        else:
            incorrect += 1
    acc = correct/(incorrect+correct)

    return acc


# loading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_data = load_data(url, 'Iris-setosa')

# create the 10 folds for cross validation
folds = create_10_folds(iris_data)

# perform 10 fold cross validation
print(ten_fold_cross_validatoin(folds))

# # visualize that the data is indeed linearly separable
# visualize_lin_sep_iris(iris_data)
#
# # train the model for the specified number of epochs
# # return trained weights and misclassified info over epochs
# iterations = 10
# w, misclassified_list = perceptron(iris_data, iterations)
#
# # display how the misclassifications vary over training
# misclass_over_its(iterations, misclassified_list)
#
# # test the model that was built on the larger data set
# test_iris_data = load_data_test(url, 100, 'Iris-setosa')
#
# #print the accuracy
# print("accuracy", perceptron_test(test_iris_data, w))