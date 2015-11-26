import time
import logging
import pickle
import pandas as pd
import numpy as np
import utilities.preProc as preProc
import utilities.classifierFuncs as cfun
from gensim.models import doc2vec

def myhash(obj):
        return hash(obj) % (2 ** 32)
    
def main():

    #Set up logging configurations
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    model = doc2vec.Doc2Vec(hashfxn=myhash)

    #Load the trained model
    model = doc2vec.Doc2Vec.load("../../classifier/Doc2VectforNLPTraining")

    word_vectors = model.syn0
      
    num_clusters = int(word_vectors.shape[0] / 5)
    # print("number of clusters: {}".format(num_clusters))
   
    print("Clustering...")
    startTime = time.time()
    cluster_index = cfun.kmeans(num_clusters, word_vectors)
    endTime = time.time()

    print("Time taken for clustering: {} minutes".format((endTime - startTime)/60))
    
    clusterf = open("../../classifier/doc2vec/clusterIndex.pickle","wb") 
    
    #Save clusters
    pickle.dump(cluster_index,clusterf)
    
    # create a word/index dictionary, mapping each vocabulary word to a cluster number
    # zip(): make an iterator that aggregates elements from each of the iterables
    index_word_map = dict(zip(model.index2word, cluster_index))
    
    train = pd.read_csv("../../data/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../../data/testData.tsv",
                   header=0, delimiter="\t", quoting=3)
    
    #Create feature vectors for training data
    trainingDataFV = np.zeros((train["review"].size, num_clusters), dtype=np.float)
    
    #Create feature vectors for test data
    testDataFV = np.zeros((test["review"].size, num_clusters), dtype=np.float)
    
    #Populate feature vectors after cleaing the data
    
    print("Processing training data...")
    counter = 0
    cleaned_training_data = preProc.clean_data(train)
    for review in cleaned_training_data:
        trainingDataFV[counter] = cfun.create_bag_of_centroids(review,num_clusters,index_word_map)
        counter += 1

    print("Processing test data...")
    counter = 0
    cleaned_test_data = preProc.clean_data(test)
    for review in cleaned_test_data:
        testDataFV[counter] = cfun.create_bag_of_centroids(review,num_clusters,index_word_map)
        counter += 1

    n_estimators = 100
    result = cfun.rfClassifer(n_estimators, trainingDataFV, train["sentiment"],testDataFV)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Doc2Vec_Clustering.csv", index=False, quoting=3)
    
if __name__ == '__main__':
    main()
