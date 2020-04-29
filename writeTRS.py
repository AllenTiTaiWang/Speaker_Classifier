import xml.etree.ElementTree as ET
import glob
import pandas as pd
import os
import process as pr

cwd = os.getcwd()
TRStest = 'TRS_test/*.trs'

def split_spkr(ids):
    """
    This functions splits the title and gets the speaker
    information.

    :param ids: id from pandas datafram
    :return: speaker
    """
    spkr = ids.split('_')[1].split('.')[0]
    return spkr

def modifyTRS(df, highpath, lowpath):
    """
    This functions writes speaker predictions into transcripts,
    and saves them in either TRS_high_con (high confident) or 
    TRS_low_con (low confident) folder, depending on their 
    predicted probability.

    :param df: Pandas dataframe with predictions
    :param highpath: The path for saving the trainscripts with
    high confident.
    :param lowpath: The path for saving the transcripts with 
    low confident.
    """
    for trs in glob.glob(os.path.join(cwd, TRStest)):
        url = trs.split('/')[-1].split('.')[0]
        xml = ET.parse(trs)
        #Match URL
        trs_df = df[df['url'] == url]
        #Tag predictions
        title = trs_df.iloc[0]['id'].split('_')[0]
        spkr1_id = trs_df.loc[trs_df['id'].apply(split_spkr) == "spkr1", 'label'].values[0]
        spkr2_id = trs_df.loc[trs_df['id'].apply(split_spkr) == "spkr2", 'label'].values[0]
        
        #Check threshould
        FailThreshould = False
       
        #Taggingi
        if spkr1_id == 1:
            spkr1 = 'coach'
        elif spkr1_id == 0:
            spkr1 = 'participant'
        else:
            spkr1 = 'spkr1'
            FailThreshould = True

        if spkr2_id == 1:
            spkr2 = 'coach'
        elif spkr2_id == 0:
            spkr2 = 'participant'
        else:
            spkr2 = 'spkr2'
            FailThreshould = True

        for spk in xml.iter('Speaker'):
            if spk.attrib['id'] == 'spkr1':
                spk.attrib['name'] = spkr1
            else:
                spk.attrib['name'] = spkr2
        
        if spkr1 == spkr2 or FailThreshould:
            xml.write(os.path.join(cwd, lowpath + '/' + title + '.trs'))
        else:
            xml.write(os.path.join(cwd, highpath + '/' + title + '.trs'))


