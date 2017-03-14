import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk

if __name__ == '__main__':

	# ************Step 1: Reading the files
    train = pd.read_csv("/Users/path/to/file/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("/Users/path/to/file/testData.tsv", header=0, delimiter="\t", quoting=3 )

    # ************Step 2: To clean the data 
    clean_train_reviews = []
    print("Cleaning and Parsing of the training set movie reviews...")

    for i in range(len(train['review'])):
    	clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train['review'][i], True)))

    # ************Step 3: Create the bag of words 
    print("Creating the bag of words...")
    vectorizer = CountVectorizer(analyzer = 'word', tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    # CountVectorizer is the scikit-learn's bag of words tool
    # The bag of words is a dictionary of frequency of all the words that occour in the goven text

    # fit_transform does two things: FIrst it fits the model and learns the vocabulary
    # second, it transforms our training data into feature vectors. The inout should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # numpy ararys are easy to work sith so er onvert it to a numpy array
    train_data_features = train_data_features.toarray()

    # ************Step 4: To create a classifier, We here are usign a random forest classifier

    print("Training the random forest classifier... (This may take time)")
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(train_data_features, train['sentiment'])

    clean_test_reviews = []


    # ************Step 5: Formating and testing of data
    print("Cleaning and pasrsing the test set movie reviews...")
    for i in range(len(test['review'])):
    	clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test['review'][i], True)))

    # creating a bag of words and covert it to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()


    # ************Step 6: Predict reviews in testing data
    print("Predicting the test labels...")
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = pd.DataFrame(data = {'id': test['id'], 'sentiment': result})
    output.to_csv("/Users/path/to/file/Bag_of_Words_model.csv", index=False, quoting=3)
    print("Wrote the results to Bag_of_Words_model.csv")