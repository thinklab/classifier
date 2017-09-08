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
    #validate_shape('ml:cases', cases, (3000,23))
    #validate_shape('ml:correct_answers', correct_answers, (3000,))
    weights = weighter(cases, correct_answers)    
    #validate_shape('weights', weights, 23)
    return lambda cases: classifier(cases, weights)

def classifier(cases, weights):
    """cases: [ncases, nfeatures]
       weights: [nfeatures]
       return: [ncases]
    """
    print("c_shape", cases.shape)
    print("w_shape", weights.shape)
    scores = np.dot(cases, weights)

    return scores > 0


def weighter(cases, correct_answers):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       return: [nfeatures]
    """
    #validate_shape('w:cases', cases, (3000,23))
    #validate_shape('w:correct_answers', correct_answers, (3000,))
    weights = np.random.normal(0, 1, [cases.shape[1]]) * 0.01
    print(weights)
    #validate_shape('w:weights_initial', weights, (23,))
    plt.ion()
    for i in range(1000000):
        print("iteration ", i)
        print("weights######################################################")
        print(weights)
        print("/weights#####################################################")        
        weights = weights_betterizer(cases, correct_answers, weights)
        #validate_shape('w:weights_iterate', weights, (23,))
        predicted_answers = classifier(cases, weights)
        print("acc - ", accuracy(predicted_answers, correct_answers))
        print("pseudo_acc - ", get_pseudo_accuracy(cases, correct_answers, weights))
        print("prec - ", precision(predicted_answers, correct_answers))
        print("recall - ", recall(predicted_answers, correct_answers))
        print("weig_len - ", np.sqrt(np.sum(weights ** 2)))
        plt.scatter(i, accuracy(predicted_answers, correct_answers))
        plt.scatter(i, get_pseudo_accuracy(cases, correct_answers, weights))
        plt.pause(0.05)
    return weights

lr = 0.001


def lr_up(signal, frame):
    global lr
    lr = lr * 2**0.25
    print('LR up; is %f now' % lr)


def lr_down(signal, frame):
    global lr
    lr = lr / 2**0.25
    print('LR down; is %f now' % lr)


import signal
signal.signal(signal.SIGUSR1, lr_up)
signal.signal(signal.SIGUSR2, lr_down)
print()


def weights_betterizer(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return weights: [nfeatures]
    """
    #validate_shape('wb:cases', cases, (3000,23))
    #validate_shape('wb:correct_answers', correct_answers, (3000,))
    function_for_action_grad = get_pseudo_accuracy_grad(cases, correct_answers, weights)
    #validate_shape('wb:function_for_action_grad', function_for_action_grad, (23,))
    #print(function_for_action_grad)
    print("grad_len - ", np.sqrt(np.sum(function_for_action_grad ** 2)))
    print("#############")
    return weights + lr * function_for_action_grad


def get_pseudo_accuracy(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return pseudo_accuracy: []
    """
    #validate_shape('gpa:cases', cases, (3000,23))
    #validate_shape('gpa:correct_answers', correct_answers, (3000,))
    #validate_shape('gpa:weights', weights, (23,))

    scores = np.dot(cases, weights)
    margin = get_margin(scores, correct_answers)
    #return np.mean(1 / (1 + np.exp(-margin)))
    return np.mean(np.minimum(1, margin))


def get_pseudo_accuracy_grad(cases, correct_answers, weights):
    """cases: [ncases, nfeatures]
       correct_answers: [ncases]
       weights: [nfeatures]
       return pseudo_accuracy_grad: [nfeatures]
    """
    scores = np.dot(cases, weights)
    print("scores#################################################### ")
    print(scores)
    margin = get_margin(scores, correct_answers)
    print(margin.shape)
    print("margin####################################################")
    print(margin)
    print("correct_answers####################################################")
    print(correct_answers)
    #da_ds = 2 * (correct_answers - 0.5) * np.exp(-margin) / ((1 + np.exp(-margin))**2)
    da_ds = 2 * (correct_answers - 0.5) * (margin < 1).astype(np.float32)
    print("da_ds#######################################################")
    print(da_ds)
    #input()
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
def validate_shape(name, value, expected_shape):
    if value.shape != expected_shape:
      raise ValueError('%r: expected shape %r, got shape %r' % (name, expected_shape, value.shape))
    else:
      print('%r: shape %r OK' % (name, value.shape))


def load_from_csv(file):
    """file: filename
       return: 
           features: np.array[ncases, nfeatures]
           descriptions: list[nfeatures]
           correct_answers: np.array[ncases]
    """
    raw_data = np.genfromtxt(file, delimiter=',', names=True)
    raw_descriptions = list(raw_data.dtype.names)
    raw_data = np.array(list(map(list, raw_data)))
    
    features = raw_data[:, 1:-1]
    descriptions = raw_descriptions[1:-1]
    correct_answers = raw_data[:,-1]  
    return features, descriptions, correct_answers

## ----------------------------------------------------------------------------
#                                   Main

def main():
    #np.seterr(invalid='ignore')
    features, descriptions, correct_answers = load_from_csv('ccard_preprocessed.csv')   
    classifier = machine_learning(features, correct_answers)
    #classifier_answers = classifier(new_cases)
    #print(classifier_answers)
if __name__ == '__main__':
    main()