import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_data(store_domain):
    base_url = store_domain + '/collections/all/products.json?page='
    data_list = []
    n_page = 1
    while True:
        page = requests.get(base_url + f'{n_page}')
        if not page.json().get('products',0):
            break
        data_list += page.json()['products']
        n_page += 1
    df = pd.DataFrame(data_list)
    return df

def transform(DF):
    df = DF.copy()
    df.drop(['vendor','product_type','published_at','created_at','updated_at','variants','images','options'],axis=1,inplace=True)
    df.body_html = df.body_html.apply(lambda x: [x.replace(' ','')])
    df.title = df.title.apply(lambda x: [i for i in x.split()])
    df.tags = df.tags.apply(lambda x: sum([i.split() for i in x],[]))
    df['soup'] = df.apply(lambda x: x['tags'] + x['title'] + x['body_html'],axis=1)
    df.soup = df.soup.apply(lambda x: set([i.lower() for i in x if len(i)>1]))
    df.soup = df.soup.apply(lambda x: ' '.join(list(x)))
    df.drop(['title','tags','body_html'],axis=1,inplace=True)
    return df

def similarity(df):
    tfidf = TfidfVectorizer(token_pattern=None,tokenizer=lambda x: x.split(' '))
    soup = tfidf.fit_transform(df.soup)
    sim_mat = cosine_similarity(soup)
    return sim_mat

def threshold(N,thresh=0.6):
    temp = list(enumerate(sim_mat[N]))
    LoT = sorted(temp,key=lambda x: x[1],reverse=True)
    new = []
    for i in range(len(LoT)):
        if LoT[i][1] >= thresh:
            new.append(LoT[i])
        else:
            break
    return sorted(new)

def return_json(df,store_domain):
    df['variations'] = df.reset_index()['index'].apply(lambda x: tuple([tup[0] for tup in threshold(x) if x-16 < tup[0] < x+16]))
    final = []
    for i in range(df.shape[0]):
        temp = set()
        for j in df.variations.iloc[i]:
            temp.update(df.variations.iloc[j])
        final.append(tuple(temp))
    
    ans = pd.Series(final).unique()
    res = []
    for alts in ans:
        res.append({'product alternates': [store_domain + '/products/' + df.handle.iloc[i] for i in alts]})
    
    return json.dumps(res)

def FindAlternateGroups(store_domain):
    DF = get_data(store_domain)
    df = transform(DF)
    sim_mat = similarity(df)
    jsonfile = return_json(df,store_domain)
    return jsonfile

FindAlternateGroups('https://www.boysnextdoor-apparel.co/')