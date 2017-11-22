
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

#convert dataset to list of sentences
print("converting dataset to list of sentences")
sentences = re.sub(r'-|\t|\n',' ',dataset)
sentences = sentences.split('.')
sentences = [sentence.translate(string.punctuation).lower().split() for sentence in sentences]
# sentences = sentences.replace('...', 'TOKEN_ELIPSES ')
# sentences = sentences.replace('.', 'TOKEN_PERIOD.')
# sentences = sentences.replace('?', 'TOKEN_QUESTION?')
# sentences = sentences.replace('"', 'TOKEN_QUOTATION')
# sentences = sentences.replace('!', 'TOKEN_EXCLAMATION!')
# sentences = sentences.replace('TOKEN_QUESTION?TOKEN_QUOTATION', 'TOKEN_QUESTIONQUOTATION')
# sentences = sentences.replace('TOKEN_EXCLAMATION!TOKEN_QUOTATION', 'TOKEN_EXCLAMATIONQUOTATION')
# sentences = sentences.replace('TOKEN_PERIOD.TOKEN_QUOTATION', 'TOKEN_PERIOD')
# sentences = re.split(r'.|\?|\!', sentences)

# 2D list to 1D list.
# sentences = [j for i in sentences for j in i]

#train word2vec
print("training word2vec")
a = time.time()
model = gensim.models.Word2Vec(sentences, min_count=5, size=ae_size, workers=4)
model.train(sentences, epochs=10, total_examples=len(sentences))
b = time.time()
print('Training time elapsed: {} s'.format(b-a))

#get most common words
print("getting common words")
dataset = [item for sublist in sentences for item in sublist]
counts = collections.Counter(dataset).most_common(500)

#reduce embeddings to 2d using tsne
print("reducing embeddings to 2D")
embeddings = np.empty((500,ae_size))
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
plt.savefig('w2v_visualization_ae250_10iter.png')

stop

model.save('w2v_model.gensim')
