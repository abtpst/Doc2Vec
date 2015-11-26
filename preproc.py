'''
Created on Sep 17, 2015

@author: atomar
'''
import pandas as pd
import nltk
import json
import logging
import utilities.preProc as preProc

if __name__ == '__main__':
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    
    train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    unlabeled_train = pd.read_csv("../../data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    num_reviews = len(train["review"])
    
    labeled = []
    
    for i in range(0, num_reviews):
        
        if( (i+1)%1000 == 0 ):
            
            print ("Labeled Review %d of %d\n" % ( i+1, num_reviews ))
        
        labeled.append(preProc.review_to_sentences(train.review[i], tokenizer,str(train.sentiment[i])))
    
    json.dump(labeled,open("../../classifier/doc2vec/labeledSentiFFF.json", "a"))
    
    unlabeled = []
        
    for i in range(0, num_reviews):
        
        if( (i+1)%1000 == 0 ):
            
            print ("Unlabeled Review %d of %d\n" % ( i+1, num_reviews ))
        
        unlabeled.append(preProc.review_to_sentences(unlabeled_train.review[i], tokenizer))    
    
    
    json.dump(unlabeled,open("../../classifier/doc2vec/unlabeledFFF.json", "a"))
