import pathlib

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.utils import shuffle

def gen_new_tuple_random_walk(data,n):
    def word_cnt(series):
        output = []
        for i in range(len(series)):
            output.append(len(str(series[i]).split()))
        return output
    new_tuples = []
    col2df = {}
    for col in list(data.columns[1:]):
        col_df = data[['id',col]]
        col_df.sort_values(by=col, key=word_cnt,inplace=True)
        col_df.reset_index(inplace=True,drop=True)
        col2df[col] = col_df
    for i_t in tqdm(range(n)):
        used_cid = set()
        new_tuple = ["sythetic_"+str(i_t)]
        for col in list(data.columns[1:]):
            id_x = random.randint(0,int(data.shape[0]*0.9))
            v_length = len(str(col2df[col][col][id_x]).split())+3
            v_length = min(v_length, len(str(col2df[col][col][int(data.shape[0]*0.9)]).split()))
            v = []
            for i in range(v_length):
                l_bound = 0
                while True:
                    id_x = random.randint(l_bound,data.shape[0]-1)
                    wrd_list =  str(col2df[col][col][id_x]).split()
                    if len(wrd_list)<i+1:
                        l_bound = id_x+1
                    else:
                        c_id = col2df[col]['id'][id_x]
                        if c_id in used_cid:
                            continue
                        else:
                            used_cid.add(c_id)
                            v.append(wrd_list[i])
                            break
            new_tuple.append(" ".join(v))
        new_tuples.append(new_tuple)
    new_df = pd.DataFrame(new_tuples,columns=data.columns)
    n_distinct_name = len(set(new_df['title'].values.tolist()))
    print("duplicate titles", new_df.shape[0]-n_distinct_name)
    data_full = pd.concat([data, new_df])
    data_full.reset_index(inplace=True, drop=True)
    return data_full


def text_variations(x, max_var = 2):
    def to_lower(x):
        return x.lower()
    def to_upper(x):
        return x.upper()
    def shuffle_words(x):
        x_wrds = x.split()
        if len(x_wrds)<=1:
            return x
        n_shuffle = random.randint(1,len(x_wrds)-1)
        for i in range(n_shuffle):
            loc_1 = random.randint(0,len(x_wrds)-1)
            loc_2 = random.randint(0,len(x_wrds)-1)
            x_wrds[loc_1], x_wrds[loc_2] =  x_wrds[loc_2],  x_wrds[loc_1]
        return " ".join(x_wrds)
    def delete_wrd(x):
        x_wrds = x.split()
        if len(x_wrds) <= 1:
            return x
        to_delete = random.randint(1, len(x_wrds) - 1)
        x_wrds = x_wrds[:to_delete]+x_wrds[to_delete+1:]
        return " ".join(x_wrds)
    all_var_funcs = [to_lower, to_upper, shuffle_words, delete_wrd]
    for _ in range(max_var):
        x = all_var_funcs[random.randint(0,len(all_var_funcs)-1)](x)
    return x

def price_variation(x):
    try:
        x = float(x)
        y = (random.random()+0.5)*x
        y = int(y)+0.99
    except:
        y = x
    return y

def get_matches(df,sample_sythetic=1.0):
    df.sort_values(by="id", inplace=True)

    cids = df['id'].values
    ids = df['_idx'].values
    i = 0
    n_match = 0
    matches = set()
    while i < len(cids):
        j = i+1
        while j < len(cids) and cids[j]==cids[i]:
            if sample_sythetic<1 and "sythetic" in cids[i]:
                if random.random()>sample_sythetic:
                    break
            j+=1
        if j-i>1:
            n_match+=(j-i)*(j-i-1)/2
            for l in range(i,j):
                for r in range(l+1,j):
                    matches.add((min(ids[l],ids[r]),max(ids[l],ids[r])))
        i = j
    print("matches:", n_match, len(matches))
    df_m = pd.DataFrame(list(matches),columns=["lid","rid"])
    return df_m

def gen_matches(data, n_matches):
    new_tuples = []
    for _ in range(n_matches):
        new_tpl = []
        base_loc = random.randint(0,data.shape[0]-1)
        for col in data.columns:
            if col == "title":
                new_tpl.append(text_variations(data[col][base_loc]))
            elif col == "price":
                new_tpl.append(price_variation(data[col][base_loc]))
            else:
                new_tpl.append(data[col][base_loc])
        new_tuples.append(new_tpl)
    new_data = pd.DataFrame(new_tuples,columns=data.columns)
    data_full = pd.concat([data, new_data])
    data_full.reset_index(inplace=True, drop=True)
    return data_full

def get_sample(data,n_sample,df_m):
    data_old1 = pd.read_csv("data/notebook_old1.csv")
    data_old2 = pd.read_csv("data/notebook_old2.csv")
    old_names_set = set(data_old1['title'].values.tolist())
    old_names_set.union(set(data_old2['title'].values.tolist()))
    match_ids_l = df_m['lid'].values.tolist()
    match_ids_r = df_m['rid'].values.tolist()
    match_ids_list_synthtic_only_l = [id for id in match_ids_l if "sythetic" in data['id'][id]]
    match_ids_list_synthtic_only_r = [id for id in match_ids_r if "sythetic" in data['id'][id]]

    #shuffle(match_ids_list_synthtic_only)
    n_syn = (n_sample-data_old1.shape[0]-data_old2.shape[0])//2
    sample_ids = match_ids_list_synthtic_only_l[:n_syn]+match_ids_list_synthtic_only_r[:n_syn]
    for i in range(data.shape[0]):
        name = data['title'][i]
        if name in old_names_set:
            sample_ids.append(i)
    print(len(sample_ids))
    sample_ids = list(set(sample_ids))
    data_sample = data.iloc[sample_ids,:]
    return data_sample

def rename_ids(data,gt):
    gt.sort_values(by="entity_id", inplace=True)
    gt.reset_index(inplace=True,drop=True)
    i = 0
    while i<gt.shape[0]:
        j = i+1
        while j<gt.shape[0] and gt['entity_id'][j]==gt['entity_id'][i]:
            data = data.replace(gt['spec_id'][j], gt['spec_id'][i], regex=False)
            j+=1
        i = j
    return data

gt = pd.read_csv("data/notebook_gt.csv")
data = pd.read_csv("data/notebook.csv")
data = rename_ids(data,gt)
data = gen_new_tuple_random_walk(data,1000000)
data = gen_matches(data,200000)
data = shuffle(data)
data.reset_index(inplace=True, drop=True)
data["_idx"] = data.index
data.fillna("nan")
df_m = get_matches(data[['_idx','id']],sample_sythetic=0.05)


data_sample = get_sample(data,2000,df_m)
df_m_sample = get_matches(data_sample[['_idx','id']])

pathlib.Path('data/sample').mkdir(parents=True, exist_ok=True)
pathlib.Path('data/secret').mkdir(parents=True, exist_ok=True)

data_sample['id'] = data_sample['_idx']
data_sample = data_sample[['id',"title"]]
data_sample.to_csv("data/sample/notebook_X.csv",index=False)
df_m_sample.to_csv("data/sample/notebook_Y.csv",index=False)

data['id'] = data['_idx']
data = data[['id',"title"]]
data.to_csv("data/secret/notebook_X.csv",index=False)
df_m.to_csv("data/secret/notebook_Y.csv",index=False)

