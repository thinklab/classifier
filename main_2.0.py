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
    function_for_action = lambda w: get_pseudo_accuracy(cases, correct_answers, w)
    function_for_action_grad = make_grad_fn(function_for_action)
    return weights + 0.01 * function_for_action_grad(weights)


def get_pseudo_accuracy(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [weights]
       return loss: []
    """
    scores = np.dot(cases, weights)
    margin = get_margin(scores, correct_answers)
    return np.mean(1 / (1 + np.exp(-margin)))

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


## ----------------------------------------------------------------------------
#                                   Main

def main():
    cases = ...
    correct_answers = ...
    new_cases = ...
    classifier = machine_learning(cases, correct_answers)
    classifier_answers = classifier(new_cases)

if __name__ == '__main__':
    main()
