
import numpy as np
import os


def label_datasets():

    targetfile = '../data/answers.csv'
    head, tail = os.path.split(targetfile)

    # filename = []
    source_labels = {}

    id2string = {0: 'Background',
                 1: 'HEU',
                 2: 'WGPu',
                 3: 'I131',
                 4: 'Co60',
                 5: 'Tc99',
                 6: 'HEUandTc99'}


    f = open(targetfile, 'r')
    a = f.readlines()
    for i in range(len(a)):
        line = a[i].strip()
        if line[0] == 'R':
            continue
        parsed = line.split(',')
        filename = parsed[0]
        source = parsed[1]
        source_time = parsed[2]
        source_labels[filename] = {'source': id2string[int(source)],
                                   'time': float(source_time)}
    f.close()
    return source_labels


def main():

    fileofinterest = '../gru/approach3_answers_crnn_11_keep0.75.csv'

    id2string = {0: 'Background',
             1: 'HEU',
             2: 'WGPu',
             3: 'I131',
             4: 'Co60',
             5: 'Tc99',
             6: 'HEUandTc99'}

    a = open(fileofinterest, 'r')
    b = a.readlines()
    b = b[1:]
    predicted = {}
    for line in b:
        raw_line = line.strip()
        parsed = raw_line.split(',')
        name = parsed[0]
        predicted[name] = {'source': id2string[int(parsed[1])],
                           'time': float(parsed[2])}
    truth = label_datasets()

    TP = 0
    FP = 0
    P = 0
    TN = 0
    FN = 0
    N = 0

    locale = 0
    locale_threshold = 10  # within 5 seconds on either side.
    counter = 0
    for item in predicted:
        locale = np.sqrt(np.power((predicted[item]['time'] - truth[item]['time']), 2))
        if predicted[item]['source'] != 'Background' and truth[item]['source'] != 'Background' and locale < locale_threshold:
            TP += 1
            P += 1
        elif predicted[item]['source'] != 'Background' and truth[item]['source'] == 'Background':
            FP += 1
            P += 1
        elif predicted[item]['source'] == 'Background' and truth[item]['source'] != 'Background':
            FN += 1
            N += 1
        else:
            TN += 1
            N += 1

        counter += 1
    print('P: {}, N: {}, T: {}\n'.format(P, N, P+N))
    # print('TPR: {}\nFPR: {}\nTNR: {}\nFNR: {}\nP: {}\nN: {}\nT: {}\n'.format(
    #     float(TP) / float(P),
    #     float(FP) / float(P),
    #     float(TN) / float(N),
    #     float(FN) / float(N),
    #     P,
    #     N,
    #     P + N))
    return

main()
