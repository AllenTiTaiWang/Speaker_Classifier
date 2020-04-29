import pandas as pd
import xml.etree.ElementTree as ET
import glob
import json
import sys
import os
cwd = os.getcwd()

#Path of all data
FidelityTable = 'fidelity_measure.xls'
TRStrain = 'TRS/*.trs'
TRSdev = 'TRS_dev/*.trs'
TRStest = 'TRS_test/*.trs'

def read_tables():
    """
    This functions read the fidelity table for matching sid to 
    each transcript.
    
    :return: A pandas table
    """
    table = pd.read_excel(os.path.join(cwd, FidelityTable))
    return table

def split_url(url):
    """
    This function splits url, and only keeps audio id.

    :param url: A string of url
    :return: Audio id
    """
    url_final = url.split("/")[-1].strip()
    return url_final

def text_to_dic(table):
    """
    The function splits transcripts by tagged speaker and saved
    them in dictionary.

    :param table: The fidelity table from `read_tables()`
    :return: A dictionary includes speaker information and conversation.
    """
    #Read transcipts
    url_dic = {}

    for transcript in glob.glob(os.path.join(cwd, TRStrain)):
        #trans_name = transcript.split("/")[-1].split(" ")[0].strip()
        url = transcript.split("/")[-1].split(" ")[-1].split(".")[0]
        #print(url)
        xml = ET.parse(transcript)
        conversation_coach = ""
        conversation_part = ""
        
        for spk in xml.iter('Speaker'):
            if spk.attrib['name'].lower() == 'coach':
                coach = spk.attrib['id']
        
        for turn in xml.iter('Turn'):
            word = ''.join(turn.itertext())
            if turn.attrib['speaker'] == coach:
                conversation_coach += word + ' '
            else:
                conversation_part += word + ' '
        #confirm url in excel
        if sum(table["call_url"].str.contains(url)) > 0:
            sid = table.loc[table["call_url"].apply(split_url) == url, "sid"].values[0]
            title_coach = sid + ' ' + url + '_coach'
            title_part = sid+ ' ' + url + '_part'
            url_dic[title_coach] = (url, conversation_coach, 1.0)
            url_dic[title_part] = (url, conversation_part, 0.0)
            #count += 1

    return url_dic

def text_to_dic_dev(table):
    """
    This functiona is pretty much the same as the previous one, except 
    that the transcripts here come from a different directory (folder).

    :param table: The fidelity table from `read_tables()`
    :return: A dictionary includes speaker information and conversation.
    """
   
    #Read transcipts
    url_dic = {}

    for transcript in glob.glob(os.path.join(cwd, TRSdev)):
        #trans_name = transcript.split("/")[-1].split(" ")[0].strip()
        url = transcript.split("/")[-1].split(" ")[-1].split(".")[0]
        #print(url)
        xml = ET.parse(transcript)
        conversation_coach = ""
        conversation_part = ""
        
        for spk in xml.iter('Speaker'):
            if spk.attrib['name'].lower() == 'coach':
                coach = spk.attrib['id']
        
        for turn in xml.iter('Turn'):
            word = ''.join(turn.itertext())
            if turn.attrib['speaker'] == coach:
                conversation_coach += word + ' '
            else:
                conversation_part += word + ' '
        #confirm url in excel
        if sum(table["call_url"].str.contains(url)) > 0:
            sid = table.loc[table["call_url"].apply(split_url) == url, "sid"].values[0]
            title_coach = sid + ' ' + url + '_coach'
            title_part = sid+ ' ' + url + '_part'
            url_dic[title_coach] = (url, conversation_coach, 1.0)
            url_dic[title_part] = (url, conversation_part, 0.0)
            #count += 1

    return url_dic


def text_to_dic_test(table):
    """
    The function splits transcripts by tagged speaker and saved
    them in dictionary. The difference between this one and previous
    two is that the data for this function doesn't have speaker 
    identity.

    :param table: The fidelity table from `read_tables()`
    :return: A dictionary includes speaker seperation and conversation.
    """
    
    #Read transcipts
    url_dic = {}
    for transcript in glob.glob(os.path.join(cwd, TRStest)):
        #trans_name = transcript.split("/")[-1].split(" ")[0].strip()
        url = transcript.split("/")[-1].split(".")[0]
        #print(url)
        xml = ET.parse(transcript)
        conversation_spkr1 = ""
        conversation_spkr2 = ""
       
        for turn in xml.iter('Turn'):
            word = ''.join(turn.itertext())
            if turn.attrib['speaker'] == 'spkr1':
                conversation_spkr1 += word + ' '
            else:
                conversation_spkr2 += word + ' '

         #confirm url in excel
        if sum(table["call_url"].str.contains(url)) > 0:
            sid = table.loc[table["call_url"].apply(split_url) == url, 'sid'].values[0]
            title_spkr1 = sid + ' ' + url + '_spkr1'
            title_spkr2 = sid+ ' ' + url + '_spkr2'
            #Give a default number
            url_dic[title_spkr1] = (url, conversation_spkr1, 0.0)
            url_dic[title_spkr2] = (url, conversation_spkr2, 0.0)
            
    return url_dic

