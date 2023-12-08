import numpy as np
import pandas as pd
import networkx as nx
import re
import random
import torch
from torch_geometric.data import Data
import time
import scipy.sparse as sp
from torch_geometric import loader
import time
import pickle
import copy

def unique(lists):  
    #delete duplicate attribute values
    lists = list(map(lambda x: x.lower(), lists ))
    if lists[0]=='':
        res = []
    else: 
        res = [lists[0]]
    for i in range(len(lists)):
        if i==0 or (lists[i] in lists[0:i]) or lists[i]=='':
            continue
        else:
            res.append(lists[i])
    return res

def construct_graph_from_df(df, G=None):
    # construct graph according to df
    if G is None:
        G = nx.Graph()
    for _, row in df.iterrows():
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)

        user_ids = row['user_mentions']
        user_ids.append(row['user_id'])
        user_ids = ['u_' + str(each) for each in user_ids]
        G.add_nodes_from(user_ids)

        words = row['filtered_words']
        words = [('w_' + each).lower() for each in words]
        G.add_nodes_from(words)

        hashtags = row['hashtags']
        hashtags = [('h_' + each).lower() for each in hashtags]
        G.add_nodes_from(hashtags)

        edges = []
        #Connect the message node with each related user node, word node, etc
        edges += [(tid, each) for each in user_ids] 
        edges += [(tid, each) for each in words]
        edges += [(tid, each) for each in hashtags]
        G.add_edges_from(edges)
    return G

def construct_graph(data,feature,index):
   #Build graph for a single tweet 
    G = nx.Graph()
    X = []

    tweet = data["text"].values
    X.append(feature[index].tolist())
    index = index+1
    tweet_id = data["tweet_id"].values
    G.add_node(tweet_id[0])

    user_loc = data["user_loc"].values

    f_w = data["filtered_words"].tolist()
    edges = []

    h_t = data["hashtags"].tolist()
    h_t = h_t[0]
    n = [user_loc[0]] + f_w[0] + h_t 
    n = unique(n)
    
    if len(n)!=0:
        for each in n:
            X.append(feature[index].tolist())
            index = index+1
        G.add_nodes_from(n)
        edges +=[(tweet_id[0], each) for each in n]
    G.add_edges_from(edges)
    return G,X

def normalize_adj(adj):
    # Symmetrically normalize adjacency matrix
    adj = sp.coo_matrix(adj) 
    rowsum = np.array(adj.sum(1)) 
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 

def aug_edge(adj): #  edge perturbation
    adj = np.array(adj)
    aug_adj1 = np.array([[i for i in j] for j in adj])
    aug_adj2 = np.array([[i for i in j] for j in adj])
    p = np.random.randint(0,len(adj)-1)
    aug_adj1[p][0] = 0
    aug_adj1[0][p] = 0
    t = np.random.randint(1,len(adj)-1)
    aug_adj1[t][p] = 1
    aug_adj1[p][t] = 1
    
    p = np.random.randint(0,len(adj)-1)
    aug_adj2[p][0] = 0
    aug_adj2[0][p] = 0
    t = np.random.randint(1,len(adj)-1)
    aug_adj2[t][p] = 1
    aug_adj2[p][t] = 1
        
    return aug_adj1,aug_adj2

def get_edge_index(adj):  #Get edge set according to adjacency matrix

    edge_index1 = []
    edge_index2 = []
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j]==1 and i<j:
                edge_index1.append(i)
                edge_index2.append(j)

    edge_index = [edge_index1] + [edge_index2]
    
    return edge_index

def get_data(message_num,start,tweet_sum):
    load_path = 'dataset/'

   # load dataset
    p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
    p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
    df_np_part1 = np.load(p_part1, allow_pickle=True)
    df_np_part2 = np.load(p_part2, allow_pickle=True)
    df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
    df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
                                           "place_type", "place_full_name", "place_country_code", "hashtags",
                                           "user_mentions", "image_urls", "entities",
                                           "words", "filtered_words", "sampled_words"])
    df = df.sort_values(by='created_at').reset_index()
    ini_df = df[start:tweet_sum]

    G = construct_graph_from_df(ini_df)
 

    combined_features = np.load(load_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')
    A = nx.adjacency_matrix(G).todense().tolist()
    
    X = []
    nodes = list(G.node)
    
    tweet=[]
    j = 0

    for i in range(len(nodes)):
        t=nodes[i][0:2]
        e=nodes[i][2:]
        if t=="t_":
            tweet.append(i)
            index=list(ini_df["tweet_id"]).index(int(e))
            X.append(list(combined_features[index]))
            j=j+1
    X = torch.tensor(X)
    adj = np.array([[0]*len(tweet)]*len(tweet))
 
    for i in range(len(tweet)):
        for j in range(len(A)):
            if A[tweet[i]][j]==1:
                for s in range(len(tweet)):
                    if A[j][tweet[s]]==1 and s!=i:
                        adj[i][s] = 1
    edge_index = get_edge_index(adj)
   
    edge_index1 = copy.deepcopy(edge_index)
    edge_index2 = copy.deepcopy(edge_index)
    true_y = torch.tensor(list(ini_df['event_id']))

    drop_percent = 0.2
    i = 0
    while 1:
        if i >= len(G.edges)*drop_percent:
            break
        m1 = random.randint(0, len(edge_index1[0])-1)
        m2 = random.randint(0, len(edge_index2[0])-1)
        if m1==m2:
            continue
        else:
            del edge_index1[0][m1]
            del edge_index1[1][m1]
            del edge_index2[0][m2]
            del edge_index2[1][m2]
       
        i = i + 1
    edge_index = torch.tensor(edge_index)
    edge_index1 = torch.tensor(edge_index1)
    edge_index2 = torch.tensor(edge_index2)

    dict_graph = {}

    dict_graph['x'] = X
    dict_graph['x1'] = X
    dict_graph['x2'] = X
    dict_graph['edge_index'] = edge_index
    dict_graph['edge_index1'] = edge_index1
    dict_graph['edge_index2'] = edge_index2
    dict_graph['y'] = true_y
    return dict_graph

def getData(M_num):  #construct an entire graph within a block
#     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    M =[20254,28976,30467,32302,34312,36146,37422,42700,44260,45623,46719,
        47951,51188,53160,56116,58665,59575,62251,64138,65537,66430,68840]  

    if M_num == 0:
        num = 0
        size = 500
    elif M_num == 1:
        num = M[M_num-1]
        size = 500
    elif M[M_num]-M[M_num-1]>2000:
        num = M[M_num-1]
        size = 1000
    else:
        num = M[M_num-1]
        size = M[M_num]-M[M_num-1]
    data = []
    i = M_num
    j = 0

    while 1:
        if (num+size)>=M[i]:
            tmp = get_data(i,num,M[i])
            data.append(tmp)
            break
        else:
            tmp = get_data(i,num,num+size)
            data.append(tmp)
            j = j + 1
            num = num+size
    return data


def get_Graph_Dataset(message_number):
    print("\nBuilding graph-level social network...")
    start_time = time.time() 
    #load data for graph-level contrastive learning
    dataset = []
    label = []
    file_name = 'dataset/GCL-data/message_block_'+str(message_number)+'.npy'
    data = np.load(file_name,allow_pickle=True)

    for dict_data in data:
        data = Data(x=dict_data['X'],x1=dict_data['x1'],x2=dict_data['x2'],
                    edge_index=dict_data['edge_index'],edge_index1=dict_data['edge_index1'],
                    edge_index2=dict_data['edge_index2'])
        dataset.append(data)
        label.append(dict_data['label'])
    if message_number == 0 :
        dataset = loader.DataLoader(dataset,batch_size=4096)
    else:
        dataset = loader.DataLoader(dataset,batch_size=len(dataset))
    end_time = time.time()
    run_time = end_time - start_time
    print("Done! It takes "+str(int(run_time))+" seconds.\n")
    return dataset,label

def get_Node_Dataset(message_number):
    #load data for node-level contrastive learning
    print("\nBuilding node-level social network...")
    start_time = time.time() 
    datas = getData(message_number)
    dataset = []
    labels = []
    
    for data in datas:
        dict_data = data

        dict_data['x'] = torch.tensor(np.array(dict_data['x']))
        dict_data['x1'] = torch.tensor(np.array(dict_data['x1']))
        dict_data['x2'] = torch.tensor(np.array(dict_data['x2']))
        dict_data['edge_index'] = torch.tensor(np.array(dict_data['edge_index']))
        dict_data['edge_index1'] = torch.tensor(np.array(dict_data['edge_index1']))
        dict_data['edge_index2'] = torch.tensor(np.array(dict_data['edge_index2']))
        data = Data(x=dict_data['x'],x1=dict_data['x1'],x2=dict_data['x2'],
                        edge_index=dict_data['edge_index'],edge_index1=dict_data['edge_index1'],
                        edge_index2=dict_data['edge_index2'])

        label = dict_data['y']
        if len(labels)==0:
            labels = label
        else:
            labels = torch.cat([labels,label])
     
        dataset.append(data)
    end_time = time.time()
    run_time = end_time - start_time
    print("Done! It takes "+str(int(run_time))+" seconds.\n")
    return dataset,np.array(labels).tolist()