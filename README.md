# Speaker Classifier

This classifier classifies speakers'utterance and adds SID to
the title of all transcripts for all transcripts in TRS file.
It will generate two folders (TRS_high_con and TRS_low_con)with
tagged transcripts. The transcripts in TRS_low_con will still
need to be manually check the tagging as it's predicted with low
confident.

This repo contains:

1. Read fidelity measures table
2. Preprocess transcripts
3. feature engineering with NLP
4. Modeling with linear machine learning models
5. Plot and save precision/recall-threshould figures
6. Evaluation on development set wiht threshould
7. Prediction on test set
8. Generate modified transcripts in two folders

## Overview of Data

One table providing the information of transcripts with their 
matched SID:
1. Fidelity measures (323 transcripts with 62 features)

## A Reminder before Getting Started

This instruction doesn't include LIVES study data. Please prepare it
before moving to the next step. And remember to put the data in the 
right position which will be specify in the following steps so that 
the scripts can spot it.

### Prerequisites

```
python>=3.72
numpy>=1.15
scikit-learn>=0.20
```

### Installing

Clone this repository in the new made directory (folder)

```
git clone https://github.com/AllenTiTaiWang/Speaker_Classifier.git
```

## Pipeline

The whole process has already built that there is only one command
to train the model, and output the modified transcripts to two folders. 
The following flow chart shows what the code will do. Note that black 
rectangles represents folders.

Specifically, we seperates the tagged model into TRS (train) and TRS_dev
(development) set, and use them to train a predictive model. The untagged 
transcripts are put into TRS_test folder. Fidelity table is required as we
need to know which SID matches which transcripts. Depending on whether their
predicted probabiility passes the threshould (confident), the transcripts 
are generated in different folders (TRS_high_con and TRS_low_con). Please
manulaay check the transcripts in TRS_low_con as it's predicted with low
confident.

![alt text](https://github.com/AllenTiTaiWang/Speaker_Classifier/blob/master/pics/flow_chart.png)

### Preparation before runnung the script

As mentioned above, the tagged and untagged transcripts should be put
into the right folder, and then we can start the process. Firstly, put
most of the tagged transcripts (at least 66% of all are recommended) in
TRS folder as train set. Remember to delete the example file int it. 
Secondly, put the rest of the tagged transcripts in TRS_dev as development 
set. Finally, the untagged transcripts should be
put in TRS_test.

### Run the script

There is only one command needed to execute it.

```
python3 main.py
```

**Threshould of Coach**, **Threshould of Participant**, **Prediction Score**,
and **Baseline Score** will be shown, and precision/recall-threshould will be
saved in pics folder as following figures. Also, the modified and tagged transcripts
will be generated in TRS_high_con and TRS_low_con files. The transcripts inTRS_low_con
file should be checked manually as it's predicted with lower confident.

![alt text](https://github.com/AllenTiTaiWang/Speaker_Classifier/blob/master/pics/plot_coach.png)
![alt text](https://github.com/AllenTiTaiWang/Speaker_Classifier/blob/master/pics/plot_participant.png)

According to the figures, and precision is more important in this task. The 
threshould is set 0.5 for being classified as a coach, and 0.1 for being 
classified as participant.

