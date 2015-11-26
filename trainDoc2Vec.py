'''
Created on Sep 22, 2015

@author: atomar
'''
from gensim.models import doc2vec
import json
import pickle
import logging
from random import shuffle
import time

#Set up logging configurations  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Load tagged cleaned up reviews
bagTaggedDocs = pickle.load(open("../../classifier/doc2vec/taggedDocs.pickle","rb"))

# parameter values
num_features = 300 #number of features/columns for the term-document matrix.

min_word_count = 40 ''' minimum word count: any word that does not occur at least this many times
 			across all documents is ignored
			'''
context = 10 # Context window size. The paper (http://arxiv.org/pdf/1405.4053v2.pdf) suggests 10 is the optimal

downsampling = 1e-3 ''' threshold for configuring which higher-frequency words are randomly downsampled;
			default is 0 (off), useful value is 1e-5
			set the same as word2vec
			'''

num_workers = 4  # Number of threads to run in parallel

# if sentence is not supplied, the model is left uninitialized
# otherwise the model is trained automatically
# https://www.codatlas.com/github.com/piskvorky/gensim/develop/gensim/models/doc2vec.py?line=192


def myhash(obj):
       return hash(obj) % (2 ** 32)
   
model = doc2vec.Doc2Vec(size=num_features,
                        window=context, min_count=min_word_count,
                        sample=downsampling, workers=4,hashfxn=myhash)

'''
python 2.x declaration would be 
model = doc2vec.Doc2Vec(size=num_features,
                        window=context, min_count=min_word_count,
                        sample=downsampling, workers=num_workers)
'''

#Build the model vocabulary (term document matrix)
model.build_vocab(bagTaggedDocs)

#Train the model for 10 epochs
for epoch in range(1,10):
    
    print("Starting Epoch ",epoch)
    
    start_time = time.time()
    
    #Shuffle the tagged cleaned up reviews in each epoch
    shuffle(bagTaggedDocs)
    
    model.train(bagTaggedDocs)
    
    print("Epoch ",epoch," took %s minutes " % ((time.time() - start_time)/60))

#Save the trained model	
model.save("../../classifier/doc2vec/Doc2VecTaggedDocs")
