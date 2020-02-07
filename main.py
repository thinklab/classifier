#!/usr/bin/env python3

import numpy as np

def  main():
    #from numpy import genfromtxt
    #features = genfromtxt('german.data', delimiter=' ')

    threshold = 0
    weights = np.array([-10, +15, -10, -6])
    features = np.array([
        # age > 70, salary > 100k, has car, has child
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 1],])
    correct_answers = np.array([
        1,
        0,
        0,
        1,
        0])
    """classifier_answers, scores = classify_many(features, threshold, weights)
    print('Classifier answers: ', classifier_answers)
    print('Accuracy: ', accuracy(classifier_answers, correct_answers))
    print('Recall: ', recall(classifier_answers, correct_answers))
    print('Precision: ', precision(classifier_answers, correct_answers))
    m = margin(scores, correct_answers)
    print('Margin: ', m)
    Lcursive = np.sum(hinge_loss(m))
    print('Hinge loss: ', m)
    print('Gradient Loss: ', np.gradient(L))"""
    minimized = train(features,correct_answers)
    print(minimized)

def margin(scores, correct_answers):
    return (correct_answers - 0.5) * 2 * scores

def hinge_loss(margin):
    return np.maximum((- margin + 1), 0)
 
def classify(features, threshold, weights):
    """
    features: [n_features]
    threshold: []
    weights: [n_features]
    """
    score = get_scores(features, weights)
    return int(score > threshold)

def classify_many(features, threshold, weights):
    """
    features: [n_objects, n_features]
    threshold: []
    weights: [n_features]
    """
    scores = get_scores(features, weights)
    return (scores > threshold).astype(int)


def get_scores(features, weights):
    return np.dot(features, weights)


def total_loss(features, correct_answers, weights):
    return np.sum(hinge_loss(margin(get_scores(features,weights),correct_answers)))

def total_loss_grad(features, correct_answers, weights):
    return nn

def train(features, correct_answers):
    """
    features: [n_objects, n_features]
    correct_answers: [n_objects]
    """
    initial_weights = np.random(features.shape[1])
    fn = lambda weights: total_loss(features, correct_answers, weights)
    fn_grad = lambda weights: total_loss_grad(features, correct_answers, weights)
    alpha = 0.001
    num_steps = 100000
    return gradient_minimize(initial_params, fn, fn_grad, alpha, num_steps)         


def accuracy(classifier_answers, correct_answers):
    """
    принимает: верные ответы, ответы классификатора
    считает и выдает: разницу между ответами в виде процента совпавших ответов

    classifier_answers: [nobjects]
    correct_answers: [nobjects]
    """
    return np.mean(np.equal(classifier_answers, correct_answers))


def precision(classifier_answers, correct_answers):
    is_executed = classifier_answers
    is_spy = correct_answers
    is_executed_spy = np.logical_and(is_executed, is_spy)
    print("vlad - ", is_executed_spy, is_executed)
    return np.sum(is_executed_spy) / np.sum(is_executed)

def recall(classifier_answers, correct_answers):
    is_executed = classifier_answers
    is_spy = correct_answers
    return np.sum(np.logical_and(is_executed, is_spy)) / np.sum(is_spy)

def specificity(classifier_answers, correct_answers):
    not_executed = 1 - classifier_answers
    not_spy = 1 - correct_answers
    return np.sum(np.logical_and(not_executed, not_spy)) / np.sum(not_spy)

def gradient_minimize(initial_params, fn, fn_grad, alpha, num_steps):
    params = initial_params
    for i in range(num_steps):
        params = params - fn_grad(params) * alpha
    return params

if __name__ == "__main__":
    main()