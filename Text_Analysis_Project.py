#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install requests')


# In[2]:


get_ipython().system('pip install beautifulsoup4')


# In[3]:


get_ipython().system('pip install nltk')


# In[4]:


get_ipython().system('pip install textstat')


# In[49]:


import pandas as pd
import requests
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from bs4 import BeautifulSoup
import textstat


# In[62]:


import nltk
nltk.download('punkt')


# In[78]:


pos_wrds = set()

with open(r'D:\Sentiment Analysis\MasterDictionary\positive-words.txt', 'r') as f:
    for l in f.readlines():
        wrd = l.strip()
        if wrd:
            pos_wrds.add(wrd)


# In[79]:


neg_wrds = set()

with open(r'D:\Sentiment Analysis\MasterDictionary\negative-words.txt', 'r') as f:
    for l in f.readlines():
        wrd = l.strip()
        if wrd:
            neg_wrds.add(wrd)


# In[80]:


stp_fls = ['StopWords_Auditor.txt', 'StopWords_Currencies.txt', 'StopWords_DatesandNumbers.txt', 'StopWords_Generic.txt', 'StopWords_GenericLong.txt', 'StopWords_Geographic.txt', 'StopWords_Names.txt']

stp_wrds = set()

for fl in stp_fls:
    with open(f'D:\Sentiment Analysis\StopWords\{fl}', 'r') as f:
        for wrd in f.readlines():
            wrd = wrd.strip()
            if wrd:
                stp_wrds.add(wrd)


# In[81]:


def txt_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article')
        
        if article:
            return article.text
        
        else:
            return None
    
    except Exception as e:
        print(f'Error extracting text from url: {e}')
        return None


# In[97]:


def cal_senti_scores(txt):
    
    wrds = word_tokenize(txt.lower())
    
    fil_wrds = []
    
    for w in wrds:
        if w.isalnum() and w not in stp_wrds:
            fil_wrds.append(w)
            
    pos_score = 0
    neg_score = 0
    
    for w in fil_wrds:            
        if w in pos_wrds:
            pos_score += 1
        elif w in neg_wrds:
            neg_score += 1
            
    polarity_sc = (pos_score - neg_score) / (pos_score + neg_score + 0.000001)   
    
    subjectivity_sc = (pos_score + neg_score) / (len(fil_wrds) + 0.000001)
    
    return pos_score, neg_score, polarity_sc, subjectivity_sc


# In[100]:


def cal_read_metrics(txt):
    
    sent = sent_tokenize(txt)   
    
    wrds = word_tokenize(txt)
    avg_sent_leng = len(wrds) / len(sent) 
    
    wrds = [w for w in word_tokenize(txt) if wrd.isalnum()]
    compl_wrd_count = 0
    
    for w in wrds:
        if textstat.syllable_count(w) >2:
            compl_wrd_count += 1
            
    per_compl_wrd = (compl_wrd_count / len(wrds)) * 100
    
    fog_index = 0.4 * (avg_sent_leng + per_compl_wrd)
    
    avg_wrds_p_sent = len(wrds) / (len(sent)) 
    
    wrd_count = len(wrds)
    
    syllable_p_wrd = sum(textstat.syllable_count(wrd) for w in wrds) / wrd_count
    
    return avg_sent_leng, per_compl_wrd, fog_index, avg_wrds_p_sent, wrd_count, syllable_p_wrd, compl_wrd_count


# In[91]:


def cal_pers_pron(txt):
    pers_pron_count = len(re.findall(r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b', txt, flags = re.IGNORECASE))
    return pers_pron_count


# In[92]:


def cal_avg_wrd_leng(txt):
    wrds = [w for w in word_tokenize(txt) if wrd.isalnum()]
    tot_leng = 0
    
    for w in wrds:
        tot_leng += len(w)
        
    avg_wrd_leng = tot_leng / len(wrds) if len(wrds) > 0 else 0
    return avg_wrd_leng


# In[93]:


input = pd.read_excel('D:\Sentiment Analysis\Input.xlsx')


# In[94]:


output = pd.DataFrame(columns = ['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])


# In[ ]:


for index, r in input.iterrows():
    url_id = r['URL_ID']
    url = r['URL']
    article_txt = txt_url(url)
    
    if article_txt:
        pos_score, neg_score, polarity_sc, subjectivity_sc = cal_senti_scores(article_txt)
        
        avg_sent_leng, per_compl_wrd, fog_index, avg_wrds_p_sent, wrd_count, syllable_p_wrd, compl_wrd_count = cal_read_metrics(article_txt)
        
        avg_wrd_leng = cal_avg_wrd_leng(article_txt)
        
        pers_pron_count = cal_pers_pron(article_txt)
        
        output = output.append({
            'URL_ID': url_id,
            'POSITIVE SCORE': pos_score,
            'NEGATIVE SCORE': neg_score,
            'POLARITY SCORE': polarity_sc,
            'SUBJECTIVITY SCORE': subjectivity_sc,
            'AVG SENTENCE LENGTH': avg_sent_leng,
            'PERCENTAGE OF COMPLEX WORDS': per_compl_wrd,
            'FOG INDEX': fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': avg_wrds_p_sent,
            'COMPLEX WORD COUNT': compl_wrd_count,
            'WORD COUNT': wrd_count,
            'SYLLABLE PER WORD': syllable_p_wrd,
            'PERSONAL PRONOUNS': pers_pron_count,
            'AVG WORD LENGTH': avg_wrd_leng
        }, ignore_index=True)

output.to_excel('output.xlsx', index=False)


# In[ ]:




