import pandas as pd
import nltk
import json
import logging
import utilities.preProc as preProc

if __name__ == '__main__':
    
    #Set up logging configurations
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    
    #Read labeled and unlabeled training data
    train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    unlabeled_train = pd.read_csv("../../data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    
    #Choose tokenizer from nltk
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    num_reviews = len(train["review"])
    
    labeled = []
    
    #Clean labeled reviews
    for i in range(0, num_reviews):
        
        if( (i+1)%1000 == 0 ):
            
            print ("Labeled Review %d of %d\n" % ( i+1, num_reviews ))
        
        #The function review_to_sentences has been defined below
        labeled.append(preProc.review_to_sentences(train.review[i], tokenizer,str(train.sentiment[i])))
    
    #Save cleaned up labeled reviews
    json.dump(labeled,open("../../classifier/doc2vec/labeledSentiFFF.json", "a"))
    
    unlabeled = []
    
    #Clean unlabeled reviews    
    for i in range(0, num_reviews):
        
        if( (i+1)%1000 == 0 ):
            
            print ("Unlabeled Review %d of %d\n" % ( i+1, num_reviews ))
        
        #The function review_to_sentences has been defined below
        unlabeled.append(preProc.review_to_sentences(unlabeled_train.review[i], tokenizer))    
    
    #Save cleaned up unlabeled reviews
    json.dump(unlabeled,open("../../classifier/doc2vec/unlabeledFFF.json", "a"))

# Here is the function review_to_sentences 

def review_to_sentences(review, tokenizer, sentiment="",removeStopwords=False, removeNumbers=False, removeSmileys=False):
    """
    This function splits a review into parsed sentences
    :param review:
    :param tokenizer:
    :param removeStopwords:
    :return: sentences, list of lists
    """
    # review.strip()remove the white spaces in the review
    # use tokenizer to separate review to sentences
    
    rawSentences = tokenizer.tokenize(review.strip())

    cleanedReview = []
    for sentence in rawSentences:
        if len(sentence) > 0:
            cleanedReview += review_to_words(sentence, removeStopwords, removeNumbers, removeSmileys)

    if(sentiment != ""):
        cleanedReview.append(sentiment)
              
    return cleanedReview

#The function review_to_words

def review_to_words(rawReview, removeStopwords=False, removeNumbers=False, removeSmileys=False):
    
    # use BeautifulSoup library to remove the HTML/XML tags (e.g., <br />)
    reviewText = BeautifulSoup(rawReview).get_text()

    # Emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replaces the pattern by the desired character/string
    
    if removeNumbers and removeSmileys:
        # any character that is not in a to z and A to Z (non text)
        reviewText = re.sub("[^a-zA-Z]", " ", reviewText)
    elif removeSmileys:
         # numbers are also included
        reviewText = re.sub("[^a-zA-Z0-9]", " ", reviewText)
    elif removeNumbers:
        reviewText = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", reviewText)
    else:
        reviewText = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", reviewText)
