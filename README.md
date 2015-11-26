# Doc2Vec

I have prepared the code base to tackle the problem of sentiment analysis. I believe that the methods that I explore here can be modified to perform document tagging. Basically, both problems involve looking at a piece of text and categorizing it. We will look at binary sentiment analysis i.e. good or bad.

The problem set is taken from

https://www.kaggle.com/c/word2vec-nlp-tutorial

The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

Doc2Vec is a neat implementation of Word2Vec. The difference here is that we can also assign tags to word vectors. This can be really useful when dealing with a multi-class tagging problem. Doc2Vec will not only give us insights into the relationships between words, but also the relationships between tags. Doc2Vec is a bit of overkill for a binary classification problem that we are exploring but let’s see how it does.

**Setup**

I am using gensim for Doc2Vec implementation. Please install using

https://radimrehurek.com/gensim/install.html

I would recommend python 3.x but it works with python2.x as well

**a.	Preprocessing**

i.	We start off by cleaning our data. This is similar to what we did for Word2Vec.

Please go through the well documented `dataCleanup.py`

ii.	However, there is a second step. We need to format the data for Doc2Vec. This is where we provide tags to the reviews. Note that it is possible to have multiple tags for a piece of text.

Please go through the well documented `prep.py`

We tag each review as LABELED_(index) or UNLABELED_(index) where index is a unique identifier for that review. Later when we are testing our model, these unique tags will allow us to identify similar reviews. Additionally, we also tag the labeled reviews with their respective sentiment tag.

**b.	Training the model**

This is similar to what we did for Word2Vec, but with a slight modification. We are going to train our model 10 times. This will give us the benefits of cross validation. Cross validation is important as it ensures that our model is not dependent of the order in which the data is fed. During each iteration, we will shuffle our data and feed it to the model. We could have done this for Word2Vec as well.

Please go through the well documented `trainDoc2Vec.py`

**c.	Testing and analyzing the model**

`testDoc2Vec.py` shows how to look into the model.

Let’s look at some of the results

`print(type(model.syn0))`
`print(model.syn0.shape)`

`<class 'numpy.ndarray'>`
`(13317, 500)`

So it’s a numpy array with 500 features and 13317 vectors. Fewer vectors than Word2Vec.

`print(model.doesnt_match("man woman child kitchen".split()) + '\n')`

`kitchen`

Ok, not bad. How about

`print(model.doesnt_match("paris berlin london austria".split()) + '\n')`

`austria`

Pretty good. How about

`print(model.most_similar(positive=['woman', 'boy'], negative=['man'], topn=10))`

`[('girl', 0.8016636967658997), ('teenage', 0.6105856895446777), ('brat', 0.6075425744056702), ('blonde', 0.5955377817153931), ('lady', 0.5924547910690308), ('orphan', 0.588573157787323), ('rich', 0.5845152735710144), ("girl's", 0.5842843651771545), ('prostitute', 0.5723811984062195), ('loretta', 0.5688214898109436)]`

Now this is not exactly what Word2Vec gave us but still, the terms make sense. Note that the score for ‘girl’ in this case is 0.80 while for Word2Vec it was 0.57. A score of 1.00 would mean a perfect match so effectively; Doc2Vec has been able to assign better scores with fewer vectors for this data set. Now let’s see if we can get any insights form the tags

`print(model.docvecs.most_similar("0"))`

`[('UNLABELED_7', 0.64825838804245), ('UNLABELED_207', 0.6387472748756409), ('UNLABELED_41', 0.631049394607544), ('UNLABELED_83', 0.6290662288665771), ('UNLABELED_55', 0.6047617793083191), ('UNLABELED_10', 0.6012109518051147), ('UNLABELED_30', 0.597069501876831), ('UNLABELED_232', 0.5969125628471375), ('UNLABELED_397', 0.5901460647583008), ('UNLABELED_59', 0.5787005424499512)]`

So we are looking for reviews similar to the ones which have the tag ‘0’ which is assigned to negative reviews. Let’s look at the review 'UNLABELED_7'

`['a', 'plane', 'carrying', 'employees', 'of', 'a', 'large', 'biotech', 'firm--including', 'the', "ceo's", 'daughter--goes', 'down', 'in', 'thick', 'forest', 'in', 'the', 'pacific', 'northwest', 'when', 'the', 'search', 'and', 'rescue', 'mission', 'is', 'called', 'off', 'the', 'ceo', 'harlan', 'knowles', '(lance', 'henriksen)', 'puts', 'together', 'a', 'small', 'ragtag', 'group', 'to', 'execute', 'their', 'own', 'search', 'and', 'rescue', 'mission', 'but', 'just', 'what', 'is', 'knowles', 'searching', 'for', 'and', 'trying', 'to', 'rescue', 'and', 'just', 'what', 'is', 'following', 'and', 'watching', 'them', 'in', 'the', 'woods', 'oy', 'what', 'a', 'mess', 'this', 'film', 'was', 'it', 'was', 'a', 'shame', 'because', 'for', 'one', 'it', 'stars', 'lance', 'henriksen', 'who', 'is', 'one', 'of', 'my', 'favorite', 'modern', 'genre', 'actors', 'and', 'two', 'it', 'could', 'have', 'easily', 'been', 'a', 'decent', 'film', 'it', 'suffers', 'from', 'two', 'major', 'flaws', 'and', "they're", 'probably', 'both', 'writer/director', 'jonas', "quastel's", 'fault--this', 'film', '(which', "i'll", 'be', 'calling', 'by', 'its', 'aka', 'of', 'sasquatch)', 'has', 'just', 'about', 'the', 'worst', 'editing', "i've", 'ever', 'seen', 'next', 'to', 'alone', 'in', 'the', 'dark', '(2005)', 'and', "quastel's", 'constant', 'advice', 'for', 'the', 'cast', 'appears', 'to', 'have', 'been', 'okay', "let's", 'try', 'that', 'again', 'but', 'this', 'time', 'i', 'want', 'everyone', 'to', 'talk', 'on', 'top', 'of', 'each', 'other', 'improvise', 'non-sequiturs', 'and', 'generally', 'try', 'to', 'be', 'as', 'annoying', 'as', 'possible', 'the', 'potential', 'was', 'there', 'despite', 'the', 'rip-off', 'aspects', '(any', 'material', 'related', 'to', 'the', 'plane', 'crash', 'was', 'obviously', 'trying', 'to', 'crib', 'the', 'blair', 'witch', 'project', '(1999)', 'and', 'any', 'material', 'related', 'to', 'the', 'titular', 'monster', 'was', 'cribbing', 'predator', '(1987))', 'ed', 'wood-like', 'exposition', 'and', 'ridiculous', 'dialogue', 'the', 'plot', 'had', 'promise', 'and', 'potential', 'for', 'subtler', 'and', 'far', 'less', 'saccharine', 'subtexts', 'the', 'monster', 'costume', 'once', 'we', 'actually', 'get', 'to', 'see', 'it', 'was', 'more', 'than', 'sufficient', 'for', 'my', 'tastes', 'the', 'mixture', 'of', 'character', 'types', 'trudging', 'through', 'the', 'woods', 'could', 'have', 'been', 'great', 'if', 'quastel', 'and', 'fellow', 'writer', 'chris', 'lanning', 'would', 'have', 'turned', 'down', 'the', 'stereotype', 'notch', 'from', '11', 'to', 'at', 'least', '5', 'and', 'spent', 'more', 'time', 'exploring', 'their', 'relationships', 'the', "monster's", 'lair', 'had', 'some', 'nice', 'production', 'design', 'specifically', 'the', 'corpse', 'decorations', 'ala', 'a', 'more', 'primitive', 'jeepers', 'creepers', '(2001)', 'if', 'it', 'had', 'been', 'edited', 'well', 'there', 'were', 'some', 'scenes', 'with', 'decent', 'dialogue', 'that', 'could', 'have', 'easily', 'been', 'effective', 'but', 'the', 'most', 'frightening', 'thing', 'about', 'sasquatch', 'is', 'the', 'number', 'of', 'missteps', 'made:', 'for', 'some', 'reason', 'quastel', 'thinks', "it's", 'a', 'good', 'idea', 'to', 'chop', 'up', 'dialogue', 'scenes', 'that', 'occur', 'within', 'minutes', 'of', 'each', 'other', 'in', 'real', 'time', 'so', 'that', 'instead', 'we', 'see', 'a', 'few', 'lines', 'of', 'scene', 'a', 'then', 'a', 'few', 'lines', 'of', 'scene', 'b', 'then', 'back', 'to', 'a', 'back', 'to', 'b', 'and', 'so', 'on', 'for', 'some', 'reason', 'he', 'thinks', "it's", 'a', 'good', 'idea', 'to', 'use', 'frequently', 'use', 'black', 'screens', 'in', 'between', 'snippets', 'of', 'dialogue', 'whether', 'we', 'need', 'the', 'idea', 'of', 'an', 'unspecified', 'amount', 'of', 'time', 'passing', 'between', 'irrelevant', 'comments', 'or', 'whether', 'the', 'irrelevant', 'comments', 'seem', 'to', 'be', 'occurring', 'one', 'after', 'the', 'other', 'in', 'time', 'anyway', 'for', 'some', 'reason', 'he', "doesn't", 'care', 'whether', 'scenes', 'were', 'shot', 'during', 'the', 'morning', 'afternoon', 'middle', 'of', 'the', 'night', 'etc', 'he', 'just', 'cuts', 'to', 'them', 'at', 'random', 'for', 'that', 'matter', 'the', 'scenes', "we're", 'shown', 'appear', 'to', 'be', 'selected', 'at', 'random', 'important', 'events', 'either', 'never', 'or', 'barely', 'appear', 'and', "we're", 'stuck', 'with', 'far', 'too', 'many', 'pointless', 'scenes', 'for', 'some', 'reason', 'he', 'left', 'a', 'scene', 'about', 'cave', 'art', 'in', 'the', 'film', 'when', 'it', 'either', 'needs', 'more', 'exposition', 'to', 'justify', 'getting', 'there', 'or', 'it', 'needs', 'to', 'just', 'be', 'cut', 'out', 'because', "it's", 'not', 'that', 'important', '(the', "monster's", 'intelligence', 'and', 'humanity', 'could', 'have', 'easily', 'been', 'shown', 'in', 'another', 'way)', 'for', 'some', 'reason', 'there', 'is', 'a', 'whole', 'character--mary', 'mancini--left', 'in', 'the', 'script', 'even', 'though', "she's", 'superfluous', 'for', 'some', 'reason', 'we', 'suddenly', 'go', 'to', 'a', 'extremely', 'soft-core', 'porno', 'scene', 'even', 'though', 'the', 'motif', 'is', 'never', 'repeated', 'again', 'for', 'some', 'reason', 'characters', 'keep', 'calling', 'harlan', 'knowles', 'mr', 'h', 'like', "they're", 'stereotypes', 'of', 'asian', 'domestics', 'for', 'some', 'reason', 'quastel', 'insists', 'on', 'using', 'the', 'blurry', 'cam', 'and', 'distorto-cam', 'for', 'the', 'monster', 'attack', 'scenes', 'even', 'though', 'the', 'costume', "doesn't", 'look', 'that', 'bad', 'and', 'it', 'would', 'have', 'been', 'much', 'more', 'effective', 'to', 'put', 'in', 'some', 'fog', 'a', 'subtle', 'filter', 'or', 'anything', 'else', 'other', 'than', 'bad', 'cinematography', 'i', 'could', 'go', 'on', 'but', 'you', 'get', 'the', 'idea', 'i', 'really', 'wanted', 'to', 'like', 'this', 'film', 'better', 'than', 'i', 'did', "i'm", 'a', 'henriksen', 'fan', "i'm", 'intrigued', 'by', 'the', 'subject', 'i', 'loved', 'the', 'setting', 'i', 'love', 'hiking', 'and', 'this', 'is', 'basically', 'a', 'hiking', 'film', 'on', 'one', 'level--but', 'i', 'just', "couldn't", 'every', 'time', 'i', 'thought', 'it', 'was', 'going', 'to', 'be', 'better', 'from', 'this', 'point', 'until', 'the', 'end', 'quastel', 'made', 'some', 'other', 'awful', 'move', 'in', 'the', 'end', 'my', 'score', 'was', 'a', '3', 'out', 'of', '10']`

So, the reviewer gave this movie a score of 3/10 which means that this review would have been classified as a 0. That means our Doc2Vec model is correct in pointing out that this particular review , is close to the reviews tagged as 0.
Again, the more data we have, the better the model will be.

**d.	Building the classifier**

Again we build a Randomforest classifier with K-means clustering.
This is exactly the same as what we did for Word2Vec but with our Doc2Vec model

`randomForestDoc2Vec.py the python script that does all of that classifierFuncs.py is the helper script for clustering and other functions`

**e.	Evaluating the classifier**
 
So this classifier does a little better than Word2Vec and gives me 84% accuracy. We can try tweaking the model parameters to see if we gain anything. 

84% accuracy on a relatively small data set is not bad but it turns out that we can do better. The real gains of Word2Vec and Doc2Vec do not appear until we go into the Big Data realm. 

**Lessons**

1.	Training the Doc2Vec model for 10 epochs seems to have made the model better at deriving relationships. This is probably the reason why Doc2Vec needed a smaller vocabulary.

2.	I noticed dramatic gains in accuracy when I decided to include numbers and smileys as valid tokens.

3.	Modifying number of features, minimum word count, context window size and number of clusters can give visible changes in accuracy.



