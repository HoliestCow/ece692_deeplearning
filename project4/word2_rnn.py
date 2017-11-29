import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import sys
import time

import gensim
import string
import re
import collections
import logging
from matplotlib import pyplot as plt

from itertools import islice

# def window(seq, n=2):
#     "Returns a sliding window (of width n) over data froxm the iterable"
#     "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
#     it = iter(seq)
#     result = tuple(islice(it, n))
#     if len(result) == n:
#         yield result
#     for elem in it:
#         result = result[1:] + (elem,)
#         yield result

# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3):

        desired_file = open('./wilde_pictureofdoriangray_tokenized.txt', 'r')
        dataset = desired_file.read()

        ### UNSUPERVISED BAG OF WORDS W2V STUFF

        # NOTE: THIS IS BUENO.
        model = gensim.models.Word2Vec.load('./1kiter_w2v_model_tokenized.gensim')

        self.w2v = model.wv

        # self.model = model

        # self.dataset = list(window(sentences, n=self.seq_len))

        #dictionary of possible characters
        # self.chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',\
                    #   '1','2','3','4','5','6','7','8','9','0','-','.',',','!','?','(',')','\'','"',' ']
        # self.num_words = len(self.chars)
        words = list(model.wv.vocab.keys())
        num_words = len(model.wv.vocab)

        #dictionary mapping characters to indices
        # self.word2index = {char:i for (i,char) in enumerate(self.chars)}
        # self.index2word = {i:char for (i,char) in enumerate(self.chars)}
        index2word = model.wv.index2word
        word2index = {}
        for i in range(len(index2word)):
            word2index[index2word[i]] = i

        self.word2index = word2index
        self.index2word = index2word

        self.n_classes = num_words
        self.n_hidden = len(model.wv[words[0]])

        # embedding_matrix = np.zeros((num_words, n_hidden), dtype=np.float32)
        # for i in range(num_words):
        #     # embedding_matrix * one_hot vector => vector representation of the word.
        #     embedding_matrix[i, :] = model.wv[words[i]]
        # embedding_matrix = tf.expand_dims(embedding_matrix, 0)

        #convert dataset to list of sentences
        print("converting dataset to list of sentences")
        sentences = re.sub(r'-|\t',' ',dataset)
        sentences = sentences.split('\n')
        empty = []
        for sentence in sentences:
            words = sentence.split()
            if words == []:
                continue
            empty += [words]
        sentences = empty
        # 2D list into a 1D list
        sentences = [j for i in sentences for j in i]

        self.data = []
        self.targets = []
        self.features = []
        self.seqlen = []
        self.vocabulary = words
        # self.embedding_matrix = embedding_matrix

        for i in range(n_samples):
            # Random sequence length
            # length = np.random.randint(min_seq_len, max_seq_len)
            length = max_seq_len
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(length)
            # Add a random or linear int sequence (50% prob)
            # if random.random() < .5:
            # Generate a linear sequence
            rand_start = np.random.randint(0, len(sentences) - length)
            s = [sentences[i] for i in
                 range(rand_start, rand_start + length)]
            # Pad sequence for dimension consistency
            # s += ['UNKNOWN' for i in range(max_seq_len - length)]
            self.data.append(s)

            # targets = s[1:]
            targets = [sentences[rand_start + length]]
            targets = self.one_hot(targets)
            targets = targets[0]
            self.targets.append(targets)  # Try to predict the next word.

            # Convert the sequence of words (vector of words) to a matrix (stacked w2v)

            # empty = np.zeros((max_seq_len - length, self.n_hidden))
            feature_matrix = np.zeros((len(s), self.n_hidden))
            for j in range(length):
                vector = model.wv[s[j]]
                feature_matrix[j, :] = vector
            # feature_matrix = np.vstack((feature_matrix, empty))
            self.features.append(feature_matrix)


            # self.labels.append([1., 0.])
            # else:
                # Generate a random sequence
                # s = [[float(random.randint(0, max_value))/max_value]
                    #  for i in range(len)]
                # Pad sequence for dimension consistency
                # s += [[0.] for i in range(max_seq_len - len)]
                # self.data.append(s)
                # self.labels.append([0., 1.])
        del model
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        # batch_labels = (self.labels[self.batch_id:min(self.batch_id +
        #                                           batch_size, len(self.data))])
        batch_features = (self.features[self.batch_id:min(self.batch_id +
                                                          batch_size, len(self.data))])
        batch_targets = (self.targets[self.batch_id:min(self.batch_id +
                                                        batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_features, batch_targets, batch_seqlen

    def one_hot(self, x):

        y = np.zeros((len(x), self.n_classes), dtype=np.int32)
        for i in range(len(x)):
            y[i, self.word2index[x[i]]] = 1.0

        return y


def dynamicRNN(x, seqlen, weights, biases, neurons):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, seq_max_len, 1)  # I Don't think this is necessary.
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.GRUCell(neurons)
    # seqlen = tf.placeholder(tf.int32, [None])
    # batch_size_T = tf.shape(x)[0]
    # lstm_cell.zero_state(batch_size_T, tf.float32)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
                                        sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    # outputs = tf.stack(outputs)
    # outputs = tf.transpose(outputs, [1, 0, 2])  # NOTE: I Don't think this is necessary.

    # Hack to build the indexing and retrieve the right output.
    # batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    # index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1) # NOTE WTF IS THIS DOING???

    # Indexing
    # outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # NOTE WTF IS THIS DOING
    outputs = outputs[:, -1, :]
    # print(weights['out'].shape)
    # print(biases['out'].shape)
    # stop

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":

    #load sample text
    # with open('wilde_pictureofdoriangray.txt','r') as f:  # my own corpus
    #     text = f.read()
    #
    # #clean up text
    # text = text.replace("\n"," ") #remove linebreaks
    # text = re.sub(' +',' ',text) #remove duplicate spaces
    # text = text.lower() #lowercase
    # Remove latex (only required for hitchikers)
    # text = text.replace("\begin{enumerate}", " ")
    # text = text.replace("\end{enumerate}", " ")
    # text = text.replace("\item ", " ")
    # text = text.replace("\begin{center}", " ")
    # text = text.replace("\end{center}", " ")
    # text = text.replace("\begin{itemize}", " ")
    # text = text.replace("\end{itemize}", " ")
    # text = text.replace("\begin{flushright}", " ")

    # ==========
    #   MODEL
    # ==========

    # Parameters
    learning_rate = 0.01
    training_steps = 1000000
    batch_size = 128
    display_step = 100
    neurons = 250

    # Network Parameters
    seq_max_len = 15 # Sequence max length

    trainset = ToySequenceData(n_samples=100000, max_seq_len=seq_max_len)
    testset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)

    n_hidden = trainset.n_hidden
    n_classes = trainset.n_classes

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, seq_max_len, n_hidden], name='batch_input')
    y = tf.placeholder(tf.int32, [None, n_classes], name='batch_targets')
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None], name='seqlen')

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = dynamicRNN(x, seqlen, weights, biases, neurons)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.999).minimize(cost)
    optimizer = tf.train.AdamOptimizer(0.02,0.9,0.999).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, training_steps + 1):
            # batch_data, batch_features, batch_targets, batch_seqlen
            batch_string, batch_x, batch_y, batch_seqlen = trainset.next(batch_size)

            # print(batch_x)
            # print(batch_x)
            # print(batch_y)
            # print(batch_y)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss, prediction = sess.run([accuracy, cost, pred], feed_dict={x: batch_x, y: batch_y,
                                                  seqlen: batch_seqlen})
                # print(np.argmax(prediction[0, :]))
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

                sample_sentence = batch_string[0]
                # print('seed')
                # print(' '.join(sample_sentence))
                sample = batch_x[0][0:15, :]
                sample = sample.reshape((1, sample.shape[0], sample.shape[1]))
                sample_seqlen = np.zeros((1,), dtype=int)
                toggle = 0
                print(sample[:, 0, 5])
                print(sample[:, -1, 5])
                for meh in range(100):
                    sample_seqlen[0] = int(sample.shape[0])
                    prediction = sess.run([pred], feed_dict={x: sample, seqlen: sample_seqlen})
                    yolo = np.argmax(softmax(prediction[0][0, :]))
                    if toggle == 0:
                        first_yolo = yolo
                        toggle = 1
                    # print(prediction[0][0, 387])
                    predicted_word = trainset.index2word[yolo]
                    predicted_word_vector = trainset.w2v[predicted_word].reshape((1, 1, len(trainset.w2v[predicted_word])))
                    sample = np.concatenate((sample, predicted_word_vector), axis=1)
                    # print(sample.shape)
                    sample = sample[:, 1:, :]
                    # print(sample.shape)
                    sample_sentence += [predicted_word]
                print(' '.join(sample_sentence))
                print(sample[:, 0, 5])
                print(sample[:, -1, 5])
                print(prediction[0][0, first_yolo])




        print("Optimization Finished!")

        # Calculate accuracy
        # test_data = testset.data
        # test_label = testset.labels
        # test_seqlen = testset.seqlen
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={x: test_data, y: test_label,
        #     seqlen: test_seqlen}))