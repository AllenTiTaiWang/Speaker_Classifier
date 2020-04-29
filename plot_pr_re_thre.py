import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_coach(y_gold, y_prob):
    """
    This plots precision/recall - threshould of coach
    figure.

    :param y_gold: 1D array of gold label
    :param y_prob: 1D array of predicted probability
    """
    plt.figure()
    precision, recall, th = precision_recall_curve(y_gold, y_prob)
    plt.plot(th, precision[1:], label="Precision",linewidth=5)
    plt.plot(th, recall[1:], label="Recall",linewidth=5)
    plt.title('Precision and recall for Coach')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    
    diff = abs(precision - recall)
    diff_ind = np.where(diff == min(diff))
    #plt.show()
    plt.savefig('pics/plot_coach.png')
    print('Threshould of Coach: ', th[diff_ind][0])

def plot_patient(y_gold, y_prob):
    """
    This plots precision/recall - threshould of coach
    figure.

    :param y_gold: 1D array of gold label
    :param y_prob: 1D array of predicted probability
    """
    plt.figure()
    y_gold = 1 - y_gold
    precision, recall, th = precision_recall_curve(y_gold, y_prob)
    plt.plot(th, precision[1:], label="Precision",linewidth=5)
    plt.plot(th, recall[1:], label="Recall",linewidth=5)
    plt.title('Precision and recall for Participant')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    
    diff = abs(precision - recall)
    diff_ind = np.where(diff == min(diff))
    #plt.show()
    plt.savefig('pics/plot_participant.png')
    print('Threshould of Participant: ', th[diff_ind][0])
