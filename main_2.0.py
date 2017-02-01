#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

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
    #print(weights)
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
    #print(weights)
    plt.ion()
    for i in range(1000):
        weights = weights_betterizer(cases, correct_answers, weights)
        predicted_answers = classifier(cases, weights)
        print("acc - ", accuracy(predicted_answers, correct_answers))
        #print("pseudo_acc - ", get_pseudo_accuracy(predicted_answers, correct_answers, weights))
        print("prec - ", precision(predicted_answers, correct_answers))
        print("recall - ", recall(predicted_answers, correct_answers))
        print("weig_len - ", np.sqrt(np.sum(weights ** 2)))
        plt.scatter(i, accuracy(predicted_answers, correct_answers), ro)
        plt.pause(0.05)
    return weights

def weights_betterizer(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return weights: [nfeatures]
    """
    function_for_action_grad = get_pseudo_accuracy_grad(cases, correct_answers, weights)
    #print(function_for_action_grad)
    print("grad_len - ", np.sqrt(np.sum(function_for_action_grad ** 2)))
    print("#############")
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
    #np.seterr(invalid='ignore')
    from numpy import genfromtxt
    train_data = genfromtxt('ccard_train.csv', delimiter = ',')
    test_data = genfromtxt('ccard_test.csv', delimiter = ',')
    correct_answers = np.array(train_data[:, -1:]) 
    #print(correct_answers)
    cases = np.array(train_data[:, 1:-1])    
    print(cases)    
    new_cases = np.array(test_data[:, 1:-1])
    #print(new_cases)
    classifier = machine_learning(cases, correct_answers)
    classifier_answers = classifier(new_cases)
    print(classifier_answers)
if __name__ == '__main__':
    main()