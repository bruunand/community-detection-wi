# Community detection
- We tried both spectral clustering and hierarchical clustering using the Girvan Newman algorithm
- The Girvan Newman algorithm was simply too slow. In practice, it took several minutes for just one betweenness
calculation

## Spectral clustering
- In order to detect the amount of clusters, we plotted the values of the second eigenvector
- This clearly showed that there were 4 communities, some severally bigger than others
![Plot of the values of the second eigenvector](community_detection/eigenvector_values.png "Eigenvector value plot")
- Given our knowledge that **k=4**, we ran the k-means clustering algorithm on the eigenvector in order to assign to
each person a cluster

## Friendship issues
- One of our biggest issues were related to pre-processing
- We were lowercasing all names, but it turns out that there are different people with the same name, although with
different casing

# Sentiment analysis
- We noticed a significant different in the sizes of the positive and negative classes.
This may have affected the results, but initially we disregarded the imbalance.
- We chose a Naive Bayes classifier as our model, mostly due to it being simple yet yielding accurate results.
Being simple, we can train it quickly and try different approaches. We can also perform cross validation more easily.

## Initial results
- Our initial results showed a **91.2%** accuracy, however with a precision of **93.23%** on the positive class and
**78.49%** on the positive class. Due to the imbalance in our dataset, we decided to use undersampling
- After balancing the dataset, we got an accuracy of **87%**. However, we still need to test which effect stemming and
negation has on the dataset

### Trial #1 scores (imbalanced dataset, 91% accuracy)
        precision    recall
    
    neg       0.78      0.64
    pos       0.93      0.96

### Trial #2 scores (balanced dataset, 87% accuracy)
        precision    recall
    
    neg       0.89      0.85
    pos       0.86      0.89
    
## Stemming
- We expect a slight decrease in accuracy when using stemming, as we risk losing the meaning of some word    
- Actually, stemming did not change much and its slightly change in accuracy may be accredited to using random
sub-sampling. Ideally, we would 

### Trial #3 scores (balanced dataset, stemming, 86% accuracy)
        precision    recall
    
    neg       0.88      0.84
    pos       0.85      0.88      

## Negation
- We expect an increase in accuracy when using negation
- For every negative word, we prepend each following word with *NEG_* until we encounter punctuation
- To our surprise, the accuracy actually decreased. The precision on the negative class is comparatively higher than the
precision on the negative class, which intuitively makes sense as the model might associate certain *NEG_* words with
a negative sentiment

## Trial #4 scores (balanced dataset, negation, 83% accuracy)
        precision    recall
    
    neg       0.86      0.79
    pos       0.80      0.87   
 
## Cross validation
- We 