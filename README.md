# Sentiment analysis
- We noticed a significant different in the sizes of the positive and negative classes.
This may have affected the results, but initially we disregarded the imbalance.
- We chose a Naive Bayes classifier as our model, mostly due to it being simple yet yielding accurate results.
Being simple, we can train it quickly and try different approaches. We can also perform cross validation more easily.

## Initial results
- Our initial results showed a **90.52%** accuracy, however with a precision of **72%** on the positive dataset and
**94%** on the positive class. Due to the imbalance in our dataset, we decided to use undersampling.
- After balancing the dataset, we got an accuracy of **86%**. However, there are still improvements to be made.
The model currently uses a count vectorizer, which counts how frequently words occur in a document. However, in large
documents common words will appear many times. To downscale the importance of these words, we tried downscaling them
with TF.


### Trial #1 scores (imbalanced dataset)
                  precision    recall  f1-score   support
    
               0       0.72      0.71      0.71      1522
               1       0.94      0.94      0.94      7616
    
       micro avg       0.91      0.91      0.91      9138
       macro avg       0.83      0.83      0.83      9138
    weighted avg       0.90      0.91      0.91      9138
    
## Trial #2 scores (balanced dataset)
                   precision    recall  f1-score   support
    
               0       0.87      0.84      0.86      1522
               1       0.84      0.88      0.86      1522
    
       micro avg       0.86      0.86      0.86      3044
       macro avg       0.86      0.86      0.86      3044
    weighted avg       0.86      0.86      0.86      3044

## TF model
- After including TF in the classification pipe, we got an accuracy of **87%**, not much better than without the
downscaling. Finally, we wanted to see if including negation of word could improve our results.

## Trial #3 (TF)
                  precision    recall  f1-score   support
    
               0       0.87      0.87      0.87      1522
               1       0.87      0.87      0.87      1522
    
       micro avg       0.87      0.87      0.87      3044
       macro avg       0.87      0.87      0.87      3044
    weighted avg       0.87      0.87      0.87      3044
    
## Negation (with TF)
- In our negation implementation, we prepend ***_neg*** to the word following a negation word.
- Example input/output:
    - Input: I am not happy today.
    - Output: I am not neg_happy neg_today.
- It seemingly made no difference as the resulting accuracy was **86%**. 

## Trial #4 (Negation)
                   precision    recall  f1-score   support
    
               0       0.86      0.86      0.86      1522
               1       0.86      0.86      0.86      1522
    
       micro avg       0.86      0.86      0.86      3044
       macro avg       0.86      0.86      0.86      3044
    weighted avg       0.86      0.86      0.86      3044