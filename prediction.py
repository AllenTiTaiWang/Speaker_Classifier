import numpy as np
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import json
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_squared_error
import os
cwd = os.getcwd()
import statistics
import sys
import pandas as pd


def split_text(text):
    """
    This functions splits the conversation, and only keep the first few
    tokens.

    :param text: A string of transcript.
    :return: the first few tokens.
    """
    text = text.split(" ") 
    cut = " ".join(text[:150])
    return cut

def dic_to_pandas(all_con):
    """
    Transform the dictionary from `text_to_dic` into pandas dataframe.
    There will be four columns, id, url, text, and label.

    :param all_con: The dictionary from `text_to_dic`
    :return: A pandas dataframe
    """
    df = pd.DataFrame.from_dict(all_con, orient='index').reset_index()
    df.columns = ['id', 'url', 'text', 'label']
    df['text'] = df['text'].apply(split_text) 
    return df

def bagofword_vectorize(x_train, x_test):
    """
    Tis function implements bag-of-word and bigram model. It transforms
    words to word count for each transcript.

    :param x_train: train set in pandas dataframe
    :param x_test: test set in pandas dataframe
    :return: Two pandas dataframe. one is training set, and the other is
    test set
    """
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    
    return x_train, x_test

def train_and_dev(train, dev):
    """
    This function trains a model and predict on dev set. The prediction, 
    gold label, and baseline prediction will be returned.

    :param train: Pandas dataframe from `dic_to_pandas` (train)
    :param dev: pandas dataframe from `dic_to_pandas` (dev)
    :return: gold label, prediction, and baseline prediction 
    """
    y_train = train['label']
    y_dev = dev['label']
    #x_train_data, x_dev_data, y_train, y_dev = train_test_split(data, label, test_size=0.25, random_state=42, stratify=data['url'])
    x_train, x_dev = bagofword_vectorize(train['text'], dev['text'])
    y_pred, y_prob = modeling(x_train, y_train, x_dev, dev)

    #Baseline
    if statistics.mean(list(y_train)) < 0.5:
        y_base = np.zeros(len(y_dev))
    else:
        y_base = np.ones(len(y_dev))

    return y_dev, y_pred, y_base, y_prob

def predict_test(train, test):
    """
    This functions train a model and predict on test set.

    :param train: dataframe from `dict_to_pandas` (train)
    :param test: dataframe from `dict_to_pandas` (test)
    :return: pandas dataframe with prediction
    """
    y_train = train['label']

    x_train, x_test = bagofword_vectorize(train['text'], test['text'])
    y_pred, y_prob = modeling(x_train, y_train, x_test, test)

    test['label'] = y_pred
    test['coa_prob'] = y_prob[:, 1]
    #print(test)
    return test

def evaluation(y_gold, y_pred, y_base):
    """
    This function evaluates the predcitions with gold label,
    and print out the accuracyscore.

    :param y_gold: gold label
    :param y_pred: predictions from the model
    :param y_base: baseline prediction
    """
    if not y_base:
        acc = accuracy_score(y_gold, y_pred)
        print("prediction score is: ", acc)
    else:
        acc_base = accuracy_score(y_gold, y_pred)
        print("baseline score is: ", acc_base)


def modeling(x_train, y_train, x_test, test):
    """
    This function builds a logistic regression model

    :param x_train: Numpy array  with features (train)
    :param y_train: Numpy array with labels
    :param x_test: Numpy array with features (test)
    :param test: Pandas dataframe
    :return: 1D numpy array
    """
    #LogisticRegression
    clf = LogisticRegression(penalty='l2', solver='liblinear', C=1000, max_iter=300)

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    pred_prob = clf.predict_proba(x_test)
    #Argmax for only one coach and one participant per transcript
    '''
    url_list = test.url.unique()
    pred_max_dev = np.array([])
    for url in url_list:
        url_index = test.index[test["url"] == url].tolist()
        #print(url_index)
        prob = clf.predict_proba(x_test[url_index, :])[:, 1]
        tmp = np.where(prob == prob[prob.argmax()], 1, 0)
        pred_max_dev = np.concatenate([pred_max_dev, tmp])
    '''
    return pred, pred_prob

def classify(y_prob, th_coach=0.5, th_part=0.1):
    '''
    This function classifies labels as coach if their probability
    is higher than 0.5, and as participant if it's lower than 0.1,
    and no prediction if it's between 0.5 and 0.1.

    :param y_prob:
    :return: 1D array for predicted label
    '''
    #print(y_prob)
    y_prob = np.where(y_prob >= 0.5, 1, y_prob)
    y_prob = np.where(y_prob <= 0.1, 0, y_prob)
    y_prob[(y_prob < 0.5) & (y_prob > 0.1)] = np.nan
    #print(y_prob)
    return y_prob
