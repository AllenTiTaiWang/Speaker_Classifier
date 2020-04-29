import argparse
import process as pr
import prediction as pred
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import os
import writeTRS as wt
import plot_pr_re_thre as pltre
import numpy as np

TRSHighConfident = 'TRS_high_con'
TRSLowConfident = 'TRS_low_con'


if __name__ == "__main__":
    """
    The main function of speaker classifier. This function
    reads the transcripts in TRS folder, and train speaker
    classifier. it validates the model on TRS_dev floder, 
    and predict the transcripts in TRS_test. The modified
    transcripts will be generated in wither TRS_low_con, or
    TRS_high_con folder, depending on its predicted 
    probability.
    """
    #Read Data
    table = pr.read_tables()
    
    #Process texts and label
    url_dic = pr.text_to_dic(table)
    url_dic_dev = pr.text_to_dic_dev(table)
    url_dic_test = pr.text_to_dic_test(table)

    #dic to pandas format
    train = pred.dic_to_pandas(url_dic)
    dev = pred.dic_to_pandas(url_dic_dev)
    test = pred.dic_to_pandas(url_dic_test)
    
    #train
    y_gold, y_pred, y_base, y_prob = pred.train_and_dev(train, dev)
    #plot precision/recall - threshould plot
    pltre.plot_coach(y_gold, y_prob[:, 1])
    pltre.plot_patient(y_gold, y_prob[:, 0])
    
    #Filter out the NA predictions
    y_pred = pred.classify(y_prob[:, 1])
    nan_array = np.isnan(y_pred)
    not_nan_array = ~ nan_array
    y_pred = y_pred[not_nan_array]
    y_gold = y_gold[not_nan_array]
    y_base = y_base[not_nan_array]
    
    #Evaluation on development
    pred.evaluation(y_gold, y_pred, False)
    #baseline
    pred.evaluation(y_gold, y_base, True)
    #Test set
    test_prediction = pred.predict_test(train, test)
    test_prediction['label'] = pred.classify(test_prediction['coa_prob'])
    
    #Output
    if not os.path.exists(TRSHighConfident):
        os.makedirs(TRSHighConfident) 
    if not os.path.exists(TRSLowConfident):
        os.makedirs(TRSLowConfident)
    wt.modifyTRS(test_prediction, TRSHighConfident, TRSLowConfident)
