import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# method that will load the data from the URL passed in as parameters
# must specify which of two class labels will be encoded as 0
def load_data(url,loc, to_zero):
    # load the data from the specified url
    data = pd.read_csv(url, header=None)
    # print the intial data
    print(data)

    # encode to_zero class label as 0
    data[loc] = np.where(data.iloc[:, -1] == to_zero, 0, 1)
    # print the data again to see the changes
    print(data)

    return data


# method that will encode all categorical to numeric
def encode_categorical(data):
    print(data)
    for header in data:
        unique = data[header].unique()
        i = 0
        for val in unique:
            data.loc[data[header] == val, header] = i
            i += 1
    print(data)


# method that will be used to create the 10 folds for cross validation
def create_10_folds(data):
    # taking a random sample of the entire dataframe ie shuffling the values randomly
    shuffleDf = data.sample(frac=1)
    # print the shuffle df showing that the new random ordering
    print(shuffleDf)
    # now that the df is such that the ordering of the samples is completely random in relation to the original ordering
    # split the df into 10 subsets that are random due to the manner in which samples were shuffled
    folds = np.array_split(shuffleDf, 10)

    return folds


# method that will perform 10 fold cross validation
def ten_fold_cross_validatoin(folds, is_iris, it):
    accum_acc = 0
    accumMetrics = [0, 0, 0, 0, 0, 0, 0]

    for i in range(len(folds)):
        # print the fold being tested
        print("Performing tests on fold:", i+1)
        # obtain the training set from all folds other than that being tested on
        train = np.concatenate((folds[:i] + folds[i + 1:]))
        # visualizing the linear separability of the training data
        if is_iris:
            visualize_lin_sep_iris(train)
        # converting the train to a np matrix to optimize operations
        train = np.asmatrix(train, dtype='float64')
        # fold i is being tested on
        test = np.asmatrix(folds[i], dtype='float64')

        # training on the training data over 10 epochs and returning the weight vector
        # as well as the misclassified list
        iterations = it
        w, misclassified_list = perceptron(train, iterations)

        # display how the misclassifications vary over training
        # can see how it varies across folds
        misclass_over_its(iterations, misclassified_list)

        # use the trained weights to predict the testing data
        confusionMxs = perceptron_test(test, w)
        metrics = print_eval(confusionMxs)

        accumInd = 0
        while accumInd < len(accumMetrics):
            accumMetrics[accumInd] = accumMetrics[accumInd] + metrics[accumInd]
            accumInd += 1
    print_accum(accumMetrics)


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

# method that will return performance measures for perceptron
# given trained weights and data to test on
def perceptron_test(data, w):
    # create confusion matrices
    rows, cols = (2, 2)
    confusionMxC1 = [[0 for i in range(cols)] for j in range(rows)]
    confusionMxC2 = [[0 for i in range(cols)] for j in range(rows)]

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

        if target == 0:
            if label.item(0, 0) == 0:
                confusionMxC1[0][0] = confusionMxC1[0][0] + 1
                confusionMxC2[1][1] = confusionMxC2[1][1] + 1
            else:
                confusionMxC1[1][0] = confusionMxC1[1][0] + 1
                confusionMxC2[0][1] = confusionMxC2[0][1] + 1
        else:
            if label.item(0, 0) == 1:
                confusionMxC1[1][1] = confusionMxC1[1][1] + 1
                confusionMxC2[0][0] = confusionMxC2[0][0] + 1
            else:
                confusionMxC1[0][1] = confusionMxC1[0][1] + 1
                confusionMxC2[1][0] = confusionMxC2[1][0] + 1

    return [confusionMxC1, confusionMxC2]


# method that will compute and print the average of the accumulated metrics
def print_accum(accumMetrics):
    accumInd = 0
    print("Averages Over 10 folds")
    labelList = ["Accuracy: ", "Micro Precision: ", 'Micro Recall: ', 'Micro F1: ',
                 "Macro Precision: ", 'Macro Recall: ', 'Macro F1: ']
    while accumInd < len(accumMetrics):
        print(labelList[accumInd], accumMetrics[accumInd]/10)
        accumInd += 1


# method that will print the evaluation statistics given a confusion matrix
def print_eval(confusionMxs):
    print(confusionMxs[0])
    acc = accuracy(confusionMxs[0])
    microPre = micro_precision(confusionMxs)
    microRe = micro_recall(confusionMxs)
    micf = micro_F1(microPre, microRe)
    macroPre = macro_precision(confusionMxs)
    macroRe = macro_recall(confusionMxs)
    macroF = macro_F1(confusionMxs)
    return [acc, microPre, microRe, micf, macroPre, macroRe, macroF]


# method that will compute the accuracy
def accuracy(confusionMx):
    print('Accuracy: ')
    tp = confusionMx[0][0]
    tn = confusionMx[1][1]
    fp = confusionMx[1][0]
    fn = confusionMx[0][1]
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(acc)
    return acc


# method that will calculate the macro_F1
def macro_F1(confusionMxs):
    print('Macro-F1:')
    precA = precision(confusionMxs[0])
    precB = precision(confusionMxs[1])
    recA = recall(confusionMxs[0])
    recB = recall(confusionMxs[1])
    F1A = F1(recA, precA)
    F1B = F1(recB, precB)
    macroF1 = (F1A + F1B)/2
    print(macroF1)
    return macroF1


# method that will calculate F1 given precision and recall values
def F1(recall, precision):
    return 2 * recall * precision / (recall + precision)


# method that will be used to calculate the macro-precision
def macro_precision(confusionMxs):
    print('Macro-Precision:')
    precA = precision(confusionMxs[0])
    precB = precision(confusionMxs[1])
    macroPre = (precA + precB)/2
    print(macroPre)
    return macroPre

#methodd that will be used for precision calculation for a confusion mx
def precision(confusionMx):
    tp = confusionMx[0][0]
    fp = confusionMx[1][0]
    if tp == fp ==0:
        return 0
    return tp/(tp+fp)

# method that will be used to calculate the macro-recall
def macro_recall(confusionMxs):
    print('Macro-Recall:')
    recA = recall(confusionMxs[0])
    recB = recall(confusionMxs[1])
    macroRec = (recA + recB)/2
    print(macroRec)
    return macroRec

#methodd that will be used for recall calculation for a confusion mx
def recall(confusionMx):
    tp = confusionMx[0][0]
    fn = confusionMx[0][1]
    if tp == fn == 0:
        return 0
    return tp/(tp+fn)

# method that will calculate the micro-precision
def micro_precision(confusionMxs):
    print('Micro-Precision:')
    tpA = confusionMxs[0][0][0]
    tpB = confusionMxs[1][0][0]
    fpA = confusionMxs[0][1][0]
    fpB = confusionMxs[1][1][0]
    microPre = (tpA + tpB)/(tpA + tpB + fpA + fpB)
    print(microPre)
    return microPre

# method that will calculate the micro-recall
def micro_recall(confusionMxs):
    print('Micro-Recall:')
    tpA = confusionMxs[0][0][0]
    tpB = confusionMxs[1][0][0]
    fnA = confusionMxs[0][0][1]
    fnB = confusionMxs[1][0][1]
    microRe = (tpA + tpB)/(tpA + tpB + fnA + fnB)
    # note that the micro-precision and the micro-recall have the same value
    print(microRe)
    return microRe

# method that will calculate the micro-F1
def micro_F1(microPre, microRe):
    print('Micro-F1:')
    # because microPre=microRe then 2*microPre*microRe/microPre+microRe = microPre/1
    microF1 = (2*microPre*microRe)/(microPre+microRe)
    print(microF1)
    return microF1


# loading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_data = load_data(url, 4,  'Iris-setosa')

# create the 10 folds for cross validation
folds = create_10_folds(iris_data)

# perform 10 fold cross validation
ten_fold_cross_validatoin(folds, True, 10)

# now also try to test the learner on the breast cancer dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data'

# load the breast cancer dataset
breast_cancer_data = load_data(url,9, 'no')

# created another method that encodes categorical attributes for this dataset
encode_categorical(breast_cancer_data)

# generate the folds
folds = create_10_folds(breast_cancer_data)

# run 10 fold cross validation with 50 epochs
ten_fold_cross_validatoin(folds,False, 50)