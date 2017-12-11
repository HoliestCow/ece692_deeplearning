
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('Predicted label')  # I've swapped these two
        plt.xlabel('True label')  # I've swapped these two

def main():
    
    # prefix = 'grudet_ep1000_lr0.001'
    prefix = 'grudetnormalsum100k'
    # prefix = 'grudet_ep50000_lr0.001'

    predictions_decode = np.load('{}_predictions.npy'.format(prefix))
    labels_decode = np.load('{}_ground_truth.npy'.format(prefix))

    class_names = ['Background',
                    'Anomaly']

    # class_names = ['Background',
    #                'HEU',
    #                'WGPu',
    #                'I131',
    #                'Co60',
    #                'Tc99',
    #                'HEUandTc99']

    cnf_matrix = confusion_matrix(predictions_decode, labels_decode)
    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          normalize=True, title='Normalized confusion matrix')
    fig.savefig('{}_classification_confusion_matrix_normalize.png'.format(prefix))

    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          normalize=False,
                          title='Unnormalized confusion matrix')
    fig.savefig('{}_classification_confusion_matrix.png'.format(prefix))
    return

main()
