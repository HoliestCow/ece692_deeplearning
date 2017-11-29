
import numpy as np
from scikitlearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def main():

    predictions_decode = np.load('predictions.npy')
    labels_decode = np.load('ground_truth.npy')
    class_names = ['Background',
                   'HEU',
                   'WGPu',
                   'I131',
                   'Co60',
                   'Tc99',
                   'HEUandTc99']

    cnf_matrix = confusion_matrix(predictions_decode, labels_decode)
    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          normalize=True, title='Normalized confusion matrix')
    fig.savefig('classification_confusion_matrix.png')
    return

main()
