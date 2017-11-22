
import numpy as np
import gensim
import string
import re
import collections
import logging
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import time

ae_size = 250

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load 20 newsgroups dataset
print("loading dataset")
# dataset = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes')).data
# dataset = ' '.join(dataset)
# dataset = unicodedata.normalize('NFKD', dataset).encode('ascii','ignore')
desired_file = open('./wilde_pictureofdoriangray.txt', 'r')
dataset = desired_file.read()
desired_file.close()

#convert dataset to list of sentences
print("converting dataset to list of sentences")
sentences = re.sub(r'-|\t|\n',' ',dataset)
# sentences = sentences.lower()
for meh in re.findall("([A-Z]+)", sentences):
    sentences = sentences.replace(meh, meh.lower())
# sentences = sentences.replace(',"', '."')  # replace the commas before quotation marks with periods.
sentences = re.sub(',"', '."', sentences)
sentences = re.sub('"', '', sentences)
sentences = re.sub('\.\.\.', '', sentences)

sentences = re.sub(',', ' COMMA', sentences) # add period token
sentences = re.sub('\.', ' PERIOD_TOKEN', sentences)
sentences = re.sub('\?', ' QUESTION_TOKEN', sentences)
sentences = re.sub('\!', ' EXCLAMATION_TOKEN', sentences)
sentences = re.split('_TOKEN', sentences)
sentences = [sentence.translate(string.punctuation).split() for sentence in sentences]

# 2D list to 1D list.
# sentences = [j for i in sentences for j in i]

#train word2vec
print("training word2vec")
a = time.time()
model = gensim.models.Word2Vec(sentences, min_count=5, size=ae_size, workers=4)
model.train(sentences, epochs=100, total_examples=len(sentences))
b = time.time()
print('Training time elapsed: {} s'.format(b-a))

## Create a modified text. ##
# sentences = [j for i in sentences for j in i]
words = list(model.wv.vocab)
ff = open('./wilde_pictureofdoriangray_tokenized.txt', 'w')
for index in range(len(sentences)):
    sentence = sentences[index]
    for word_index in range(len(sentence)):
        word = sentence[word_index]
        if word not in words:
            sentence[word_index] = 'UNKNOWN'
    ff.write(' '.join(sentence))
    ff.write('\n')
ff.close()

print("loading dataset")
# dataset = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes')).data
# dataset = ' '.join(dataset)
# dataset = unicodedata.normalize('NFKD', dataset).encode('ascii','ignore')
desired_file = open('./wilde_pictureofdoriangray_tokenized.txt', 'r')
dataset = desired_file.read()
desired_file.close()

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

#train word2vec
print("training word2vec")
a = time.time()
model = gensim.models.Word2Vec(sentences, min_count=5, size=ae_size, workers=4)
model.train(sentences, epochs=100, total_examples=len(sentences))
b = time.time()
print('Training time elapsed: {} s'.format(b-a))

#get most common words
print("getting common words")
dataset = [item for sublist in sentences for item in sublist]
counts = collections.Counter(dataset).most_common(500)

#reduce embeddings to 2d using tsne
print("reducing embeddings to 2D")
embeddings = np.empty((500, ae_size))
for i in range(500):
    embeddings[i,:] = model[counts[i][0]]
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
embeddings = tsne.fit_transform(embeddings)

#plot embeddings
print("plotting most common words")
fig, ax = plt.subplots(figsize=(30, 30))
for i in range(500):
    ax.scatter(embeddings[i,0],embeddings[i,1])
    ax.annotate(counts[i][0], (embeddings[i,0],embeddings[i,1]))

#save to disk
plt.savefig('w2v_visualization_1kiter_tokenized.png')

model.save('1kiter_w2v_model_tokenized.gensim')
# stop

# model = gensim.models.Word2Vec.load('./10kiter_w2v_model.gensim')
#
# limit = 25
# sentence_number = 50
#
# index2word = model.wv.index2word
# word2index = {}
# for i in range(len(index2word)):
#     word2index[index2word[i]] = i

# Sentence Generation
# print(sentences)
# num_words = len(model.wv.vocab)
# max_words = 60
# print(num_words)
# for i in range(sentence_number):
#     index = np.random.randint(num_words)
#     string = [model.wv.index2word[index]]
#     # while True:
#     for j in range(max_words):
#         maybe = model.predict_output_word(string, topn=limit)
#         # print(maybe)
#         # randomizer = np.random.randint(len(maybe))
#         for j in range(limit):
#             if maybe[j][0] in string:
#                 continue
#             else:
#                 string += [maybe[j][0]]
#                 break
#             if (maybe[j][0] == 'PERIOD' or maybe[j][0] == 'QUESTION' or maybe[j][0] == 'EXCLAMATION'):
#                 break
#         if ('PERIOD' in string or 'QUESTION' in string or 'EXCLAMATION' in string):
#             print(' '.join(string))
#             break
#         if j >= limit:
#             print(' '.join(string))
#             break
#     print(' '.join(string))
