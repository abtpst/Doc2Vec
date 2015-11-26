'''
Created on Sep 23, 2015

@author: atomar
'''
import pickle
import json
from gensim.models.doc2vec import TaggedDocument


def labelizeReviews(reviewSet,labelType):
        """
        add label to each review
        :param reviewSet:
        :param label: the label to be put on the review
        :return:
        """
        labelized = []
        for index, review in enumerate(reviewSet):
            
            if(labelType == "LABELED"):
                
                sentiment = review.pop()
                labelized.append(TaggedDocument(words=review, tags=['%s_%s'%(labelType, index),sentiment]))
            
            else:
            
                labelized.append(TaggedDocument(words=review, tags=['%s_%s'%(labelType, index)]))
              
        return labelized
   
labeled = json.load(open("../../classifier/doc2vec/labeledSentiFFF.json", "r"))

unlabeled = json.load(open("../../classifier/doc2vec/unlabeledFFF.json", "r"))
   
labeled = labelizeReviews(labeled, 'LABELED')

unlabeled = labelizeReviews(unlabeled, 'UNLABELED')
    
bagTaggedDocs = labeled + unlabeled

pickle.dump(bagTaggedDocs,open("../../classifier/doc2vec/taggedDocs.pickle", "wb"))
