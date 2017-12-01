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

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

class character_rnn(object):
    '''
    sample character-level RNN by Shang Gao

    parameters:
      - seq_len: integer (default: 200)
        number of characters in input sequence
      - first_read: integer (default: 50)
        number of characters to first read before attempting to predict next character
      - rnn_size: integer (default: 200)
        number of rnn cells

    methods:
      - train(text,iterations=100000)
        train network on given text
    '''
    def __init__(self, seq_len=30, first_read=15, rnn_size=100, ae_size=100):

        model = gensim.models.Word2Vec.load('./w2v_model.gensim')

        desired_file = open('./wilde_pictureofdoriangray.txt', 'r')
        dataset = desired_file.read()

        #convert dataset to list of sentences
        print("converting dataset to list of sentences")
        sentences = re.sub(r'-|\t|\n',' ',dataset)
        sentences = sentences.split('.')
        sentences = [sentence.translate(string.punctuation).lower().split() for sentence in sentences]

        self.sentences = [j for i in sentences for j in i]

        self.seq_len = seq_len
        self.first_read = first_read

        self.dataset = list(window(sentences, n=self.seq_len))

        #dictionary of possible characters
        # self.chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',\
                    #   '1','2','3','4','5','6','7','8','9','0','-','.',',','!','?','(',')','\'','"',' ']
        # self.num_words = len(self.chars)
        self.words = model.wv.vocab
        self.num_words = len(model.wv.vocab)

        #dictionary mapping characters to indices
        # self.word2index = {char:i for (i,char) in enumerate(self.chars)}
        # self.index2word = {i:char for (i,char) in enumerate(self.chars)}
        self.index2word = model.wv.index2word
        self.word2index = {}
        for i in range(len(self.index2word)):
            self.word2index[self.index2word[i]] = i
        '''
        #training portion of language model
        '''

        #input sequence of character indices
        self.input = tf.placeholder(tf.int32,[1, seq_len])

        #convert to one hot
        one_hot = tf.one_hot(self.input, self.num_words)

        #rnn layer
        self.gru = GRUCell(rnn_size)
        outputs, states = tf.nn.dynamic_rnn(self.gru, one_hot, sequence_length=[seq_len],dtype=tf.float32)
        outputs = tf.squeeze(outputs,[0])

        #ignore all outputs during first read steps
        outputs = outputs[first_read:-1]

        #softmax logit to predict next character (actual softmax is applied in cross entropy function)
        logits = tf.layers.dense(outputs,self.num_words,None,True,tf.orthogonal_initializer(),name='dense')

        #target character at each step (after first read chars) is following character
        targets = one_hot[0, first_read+1:]

        #loss and train functions
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=targets))
        self.optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.999).minimize(self.loss)

        '''
        #generation portion of language model
        '''

        #use output and state from last word in training sequence
        state = tf.expand_dims(states[-1],0)
        output = one_hot[:,-1]

        #save predicted characters to list
        self.predictions = []

        #generate 100 new characters that come after input sequence
        for i in range(100):

            #run GRU cell and softmax
            output,state = self.gru(output,state)
            logits = tf.layers.dense(output,self.num_words,None,True,tf.orthogonal_initializer(),name='dense',reuse=True)

            #get index of most probable character
            output = tf.argmax(tf.nn.softmax(logits),1)

            #save predicted character to list
            self.predictions.append(output)

            #one hot and cast to float for GRU API
            output = tf.cast(tf.one_hot(output,self.num_words),tf.float32)

        #init op
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # def train(self,text,iterations=100000):
    def train(self, iterations=100000):
        '''
        train network on given text

        parameters:
          - text: string
            string to train network on
          - iterations: int (default: 100000)
            number of iterations to train for

        outputs:
            None
        '''
        #convert characters to indices
        # print("converting text in indices")
        # print(text)
        # sentences = re.sub(r'-|\t|\n',' ', text)
        # sentences = sentences.s;plit('.')
        # sentences = [sentence.translate(string.punctuation).lower().split() for sentence in sentences]
        # sentences = [j for i in sentences for j in i]
        # self.dataset = list(window(sentences, n=self.seq_len))
        text = self.sentences
        # sentences = [j for i in sentences for j in i]
        text_indices = [self.word2index[char] for char in text if char in self.word2index]

        #get length of text
        text_len = len(text_indices)

        #train
        for i in range(iterations):

            #select random starting point in text
            start = np.random.randint(text_len - self.seq_len)
            sequence = text_indices[start:start+self.seq_len]

            string = []
            for item in sequence:
                string += [self.index2word[item]]

            #train
            feed_dict = {self.input:[sequence]}
            loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
            # sys.stdout.write("iterations %i loss: %f  \r" % (i+1,loss))
            # sys.stdout.flush()

            #show generated sample every 100 iterations
            if (i+1) % 100 == 0:

                feed_dict = {self.input:[sequence]}
                pred = self.sess.run(self.predictions, feed_dict=feed_dict)
                sample = ' '.join([self.index2word[idx[0]] for idx in pred])
                print('iteration {} generated loss: {}, text sample: {}'.format(i+1, loss, sample))
                # sys.stdout.flush()
            if (i+1) % 1000 == 0:
                sys.stdout.flush()


if __name__ == "__main__":

    import re

    #load sample text
    with open('wilde_pictureofdoriangray.txt','r') as f:  # my own corpus
        text = f.read()

    #clean up text
    text = text.replace("\n"," ") #remove linebreaks
    text = re.sub(' +',' ',text) #remove duplicate spaces
    text = text.lower() #lowercase
    # Remove latex (only required for hitchikers)
    # text = text.replace("\begin{enumerate}", " ")
    # text = text.replace("\end{enumerate}", " ")
    # text = text.replace("\item ", " ")
    # text = text.replace("\begin{center}", " ")
    # text = text.replace("\end{center}", " ")
    # text = text.replace("\begin{itemize}", " ")
    # text = text.replace("\end{itemize}", " ")
    # text = text.replace("\begin{flushright}", " ")

    #train rnn
    rnn = character_rnn(seq_len=30, first_read=15, rnn_size=30, ae_size=500)
    a = time.time()
    rnn.train(iterations=100000000)
    b = time.time()
    print('Training time elapsed: {}'.format(b-a))

