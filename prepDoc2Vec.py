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
            
            #We give two lables to each review. The 'LABELED/UNLABELED_INDEX' and the sentiment
            if(labelType == "LABELED"):
                
                sentiment = review.pop()
                
                labelized.append(TaggedDocument(words=review, tags=['%s_%s'%(labelType, index),sentiment]))
            
            else:
                labelized.append(TaggedDocument(words=review, tags=['%s_%s'%(labelType, index)]))
              
        return labelized

#Load cleaned up labeled reviews   
labeled = json.load(open("../../classifier/doc2vec/labeledSentiFFF.json", "r"))

#Load cleaned up unlabeled reviews
unlabeled = json.load(open("../../classifier/doc2vec/unlabeledFFF.json", "r"))

#Tag the cleaned labeled reviews   
labeled = labelizeReviews(labeled, 'LABELED')

#Tag the cleaned unlabeled reviews
unlabeled = labelizeReviews(unlabeled, 'UNLABELED')

#Combine the tagged reviews    
bagTaggedDocs = labeled + unlabeled

#Save tagged reviews
pickle.dump(bagTaggedDocs,open("../../classifier/doc2vec/taggedDocs.pickle", "wb"))
