#!/usr/bin/env python3
"""
    Gradient descent. Sample dataset is here:

    https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients


"""
import glob
import sys

import os

import re
import signal
import numpy as np
import matplotlib.pyplot as plt

DELIMITER = '=' * 80
#
# ----------------------------------------------------------------------------
#                          Quality measurement


def accuracy(predicted_answers, correct_answers):
    """predicted_answers: [ncases]
       correct_answers: [ncases]
       return: []
    """
    return np.mean(np.equal(predicted_answers, correct_answers))


def precision(predicted_answers, correct_answers):
    """predicted_answers: [ncases]
       correct_answers: [ncases]
       return: []
    """
    executed_spies = np.dot(predicted_answers, correct_answers)
    executed = np.sum(predicted_answers)
    return executed_spies / executed


def recall(predicted_answers, correct_answers):
    """predicted_answers: [ncases]
       correct_answers: [ncases]
       return: []
    """
    executed_spies = np.dot(predicted_answers, correct_answers)
    count_of_spies = np.sum(correct_answers)
    return executed_spies / count_of_spies

#
# ------------------------------------------------------------------------------
#                            LEARNING RATE UP DOWN
# send signal to process pid when running:


LR = 0.001


def lr_up(signal, frame):
    """
    usage: kill -s SIGUSR1 [pid]
    """
    global LR  # pylint: disable=W0603
    LR = LR * 2 ** 0.25
    print('LR up; is %f now' % LR)


def lr_down(signal, frame):
    """
    usage: kill -s SIGUSR2 [pid]
    """
    global LR  # pylint: disable=W0603
    LR = LR / 2 ** 0.25
    print('LR down; is %f now' % LR)


signal.signal(signal.SIGUSR1, lr_up)
signal.signal(signal.SIGUSR2, lr_down)
print()

#
# ------------------------------------------------------------------------------
#                            The MACHINE LEARNING


def get_latest_weights_file():
    """return: filename: str
       iterations: int
    """

    def extract_number(name):
        s_name = re.findall("\d+", name)
        return int(s_name[0]) if s_name else -1, name

    files = glob.glob('*.*npy')
    if not files:
        return None
    latest_file = max(files, key=extract_number)
    iterations = extract_number(latest_file)[0]
    return latest_file, iterations


def machine_learning(cases, correct_answers):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       return: classifier

       def classifier(cases):
           cases: [ncases, nfeatures]
           return: [ncases]
    """
    # validate_shape('ml:cases', cases, (3000,23))
    # validate_shape('ml:correct_answers', correct_answers, (3000,))
    weights = weighter(cases, correct_answers)
    # validate_shape('weights', weights, 23)
    return lambda cases: classifier(cases, weights)


def classifier(cases, weights):
    """cases: [ncases, nfeatures]
       weights: [nfeatures]
       return: [ncases]
    """
    scores = np.dot(cases, weights)

    return scores > 0


def weighter(cases, correct_answers):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       return: [nfeatures]
    """
    # validate_shape('w:cases', cases, (3000,23))
    # validate_shape('w:correct_answers', correct_answers, (3000,))
    weights_file, iterations = get_latest_weights_file()
    if weights_file:
        weights = np.load(weights_file)
        i = iterations
    else:
        weights = np.random.normal(0, 1, [cases.shape[1]]) * 0.01
        i = 0
    plt.ion()
    pid = os.getpid()
    try:
        for i in range(i, 1000000):
            print(DELIMITER)
            print(DELIMITER)
            print('pid {}'.format(pid))
            print('iteration ', i)
            print(DELIMITER)
            print('weights')
            print(weights)
            print('/weights')
            print(DELIMITER)
            weights = weights_betterizer(cases, correct_answers, weights)
            # validate_shape('w:weights_iterate', weights, (23,))
            predicted_answers = classifier(cases, weights)
            acc = accuracy(predicted_answers, correct_answers)
            print("acc - ", acc)
            pseudo_acc = get_pseudo_accuracy(cases, correct_answers, weights)
            print("pseudo_acc - ", pseudo_acc)
            prec = precision(predicted_answers, correct_answers)
            print("prec - ", prec)
            print("recall - ", recall(predicted_answers, correct_answers))
            print("weig_len - ", np.sqrt(np.sum(weights ** 2)))
            plt.scatter(i, acc)
            plt.scatter(i, pseudo_acc)
            plt.pause(0.05)
    except KeyboardInterrupt:
        filename = ('weights-{}'.format(i))
        np.save(filename, arr=weights)
        print(DELIMITER)
        print('Saved new weights to {}.npy'.format(filename))
        print(DELIMITER)
        sys.exit()
    return weights


def weights_betterizer(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return weights: [nfeatures]
    """
    # validate_shape('wb:cases', cases, (3000,23))
    # validate_shape('wb:correct_answers', correct_answers, (3000,))
    function_for_action_grad = get_pseudo_accuracy_grad(cases, correct_answers,
                                                        weights)
    # validate_shape('wb:function_for_action_grad',
    # function_for_action_grad, (23,))
    # print(function_for_action_grad)
    print("grad_len - ", np.sqrt(np.sum(function_for_action_grad ** 2)))
    print(DELIMITER)
    return weights + LR * function_for_action_grad


def get_pseudo_accuracy(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return pseudo_accuracy: []
    """
    # validate_shape('gpa:cases', cases, (3000,23))
    # validate_shape('gpa:correct_answers', correct_answers, (3000,))
    # validate_shape('gpa:weights', weights, (23,))

    scores = np.dot(cases, weights)
    margin = get_margin(scores, correct_answers)
    # return np.mean(1 / (1 + np.exp(-margin)))
    return np.mean(np.minimum(1, margin))


def get_pseudo_accuracy_grad(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return pseudo_accuracy_grad: [nfeatures]
    """
    scores = np.dot(cases, weights)
    print("scores")
    print(scores)

    margin = get_margin(scores, correct_answers)
    print("margin")
    print(margin)

    print("correct_answers")
    print(correct_answers)

    # da_ds = 2 * (correct_answers - 0.5) * np.exp(-margin) /
    # ((1 + np.exp(-margin))**2)
    da_ds = 2 * (correct_answers - 0.5) * (margin < 1).astype(np.float32)
    print("da_ds")
    print(da_ds)
    print(DELIMITER)
    return np.dot(da_ds, cases) / cases.shape[0]


def get_margin(scores, correct_answers):
    """
        scores: [ncases]
        correct_answers: [ncases]
        return margin: [ncases]
    """
    return scores * 2 * (correct_answers - 0.5)


def make_grad_fn(function):
    """
    def fn(x):
        x: [n]
        return: []

    return: grad_fn

    def grad_fn(x):
        x: [n]
        return: [n]
    """
    raise NotImplementedError()


def validate_shape(name, value, expected_shape):
    """
    name: string
    value: np.array
    expected_shape: tuple
    """
    if value.shape != expected_shape:
        raise ValueError('%r: expected shape %r, got shape %r' % (
            name, expected_shape, value.shape))
    print('%r: shape %r OK' % (name, value.shape))


def load_from_csv(file):
    """file: filename
       return: features: np.array[ncases, nfeatures]
               descriptions: list[nfeatures]
               correct_answers: np.array[ncases]
    """
    raw_data = np.genfromtxt(file, delimiter=',', names=True)
    raw_descriptions = list(raw_data.dtype.names)
    raw_data = np.array(list(map(list, raw_data)))

    raw_features = raw_data[:, 1:-1]
    descriptions = raw_descriptions[1:-1]
    raw_correct_answers = raw_data[:, -1]

    test_dataset_size = int(raw_data.shape[0] * 0.8)  # pylint: disable=E1136

    learn_features = raw_features[:test_dataset_size, :]
    learn_correct_answers = raw_correct_answers[:test_dataset_size]

    test_features = raw_features[test_dataset_size:, :]
    test_correct_answers = raw_correct_answers[test_dataset_size:]
    return learn_features, learn_correct_answers, \
        test_features, test_correct_answers, descriptions


# ----------------------------------------------------------------------------
#                                   Main

def main():
    """
    input: preprocessed data for training or previously betterized weights
    """
    # np.seterr(invalid='ignore')
    learn_features, learn_correct_answers, \
        test_features, test_correct_answers, descriptions = \
        load_from_csv('ccard_preprocessed.csv')

    classifier_ = machine_learning(learn_features, learn_correct_answers)
    classifier_answers = classifier_(test_features)
    print(classifier_answers, test_correct_answers)


if __name__ == '__main__':
    main()
