import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import train_test_split


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
    raw_correct_answers = raw_data[:, -1].astype(int)
    learn_features, test_features, \
        learn_correct_answers, test_correct_answers = train_test_split(
            raw_features, raw_correct_answers, test_size=0.33, random_state=42)
    return learn_features, learn_correct_answers, \
        test_features, test_correct_answers, descriptions


def main():
    """
    input: preprocessed data for training or previously betterized weights
    """
    # np.seterr(invalid='ignore')
    learn_features, learn_correct_answers, \
        test_features, test_correct_answers, descriptions = \
        load_from_csv('../preprocessing/ccard_preprocessed.csv')

    classifier = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000000)
    classifier.fit(learn_features, learn_correct_answers)

    average_precision = average_precision_score(test_features, test_correct_answers)

    disp = plot_precision_recall_curve(classifier, test_features, test_correct_answers)

    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))


if __name__ == '__main__':
    main()

