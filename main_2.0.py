#!/usr/bin/env python3

import numpy as np

## ----------------------------------------------------------------------------
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

## ----------------------------------------------------------------------------
#                            The MACHINE LEARNING

def machine_learning(cases, correct_answers):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       return: classifier

       def classifier(cases):
           cases: [ncases, nfeatures]
           return: [ncases]
    """
    weights = weighter(cases, correct_answers)
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
    weights = np.random.normal(0, 1, [cases.shape[1]])
    for i in range(10000):
        weights = weights_betterizer(cases, correct_answers, weights)
    return weights

def weights_betterizer(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return weights: [nfeatures]
    """
    function_for_action_grad = get_pseudo_accuracy_grad(cases, correct_answers, weights)
    return weights + 0.01 * function_for_action_grad

def get_pseudo_accuracy(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return pseudo_accuracy: []
    """
    scores = np.dot(cases, weights)
    margin = get_margin(scores, correct_answers)
    return np.mean(1 / (1 + np.exp(-margin)))

def get_pseudo_accuracy_grad(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return pseudo_accuracy_grad: [nfeatures]
    """
    scores = np.dot(cases, weights)
    margin = get_margin(scores, correct_answers)
    da_ds = 2 * (correct_answers - 0.5) * np.exp(-margin) / (1 + np.exp(-margin))**2
    return np.dot(da_ds, cases) / cases.shape[0]

def get_margin(scores, correct_answers):
    """
        scores: [ncases]
        correct_answers: [ncases]
        return margin: [ncases]
    """
    return scores * 2 * (correct_answers - 0.5)

def make_grad_fn(fn):
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

## ----------------------------------------------------------------------------
#                                   Main

def main():
    from numpy import genfromtxt
    train_data = genfromtxt('training.csv', delimiter = ',')
    test_data = genfromtxt('test.csv', delimiter = ',')
    cases = np.delete(train_data, 1, 1)    
    #print(cases)
    correct_answers = np.array(train_data[:, 1]) 
    #print(correct_answers)
    new_cases = np.delete(test_data, 1, 1)
    classifier = machine_learning(cases, correct_answers)
    classifier_answers = classifier(new_cases)

if __name__ == '__main__':
    main()