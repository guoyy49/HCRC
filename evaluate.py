from singlepass import SinglePass
import numpy as np
import os 
import time
import random
from easy_drl.policy.ppo import PPO

from sklearn.metrics import normalized_mutual_info_score,adjusted_mutual_info_score,adjusted_rand_score,silhouette_score

M =[20254,28976,30467,32302,34312,36146,37422,42700,44260,45623,46719,
    47951,51188,53160,56116,58665,59575,62251,64138,65537,66430,68840]
def DRL_cluster(all_embeddings,block_num,pred_label):
    para = 0.1
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, para, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
    else:
        print("Using DRL-Single-Pass to learn threshold...")
        global_step = 0
        agent = PPO([5], 1, continuous=True)
        sp_sim = SinglePass(0.6, all_embeddings, 1, pred_label, M[block_num] - M[block_num - 1], agent, para, M[block_num-1]-2000, sim=True)
        
        global_step = sp_sim.global_step
        sp = SinglePass(0.6, all_embeddings, 1, pred_label, M[block_num] - M[block_num - 1], agent, para, M[block_num-1]-2000, sim=False)
    
    return sp.cluster_result,sp.sim_threshold
    
def random_cluster(all_embeddings,block_num,pred_label):
    threshold = random.uniform(0.6,0.8)
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = 0.87
    else:
        print("Evaluating message block...")
        start_time = time.time()
        sp = SinglePass(threshold, all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
    return sp.cluster_result,threshold
    
def semi_cluster(all_embeddings,label,block_num,pred_label):
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = 0.87
    else:
        print("Evaluating message block...")
        start_time = time.time()
        embeddings = all_embeddings.tolist()
        size = M[block_num] - M[block_num - 1]
        embeddings = embeddings[0:len(embeddings)-int(size*0.9)]
        pre_label = pred_label[0:len(embeddings)]
        
        size = len(embeddings) - M[block_num - 1]
        embeddings = np.array(embeddings)
        thresholds = [0.6,0.65,0.7,0.75,0.8]
        s1s = []
        for t in thresholds:
            sp = SinglePass(t, embeddings, 2, pre_label, size, None, 0, 0, sim=False)
            true_label = label[0:len(sp.cluster_result)]
            s1 = normalized_mutual_info_score(true_label, sp.cluster_result, average_method='arithmetic')
            s1s.append(s1)
        index = s1s.index(max(s1s))
        sp = SinglePass(thresholds[index], all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = thresholds[index]
    return sp.cluster_result,threshold
        
def NMI_cluster(all_embeddings,label,block_num,pred_label):
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = 0.87
    else:
        print("Evaluating message block...")
        start_time = time.time()
        thresholds = [0.6,0.65,0.7,0.75,0.8]
        s1s = []
        for t in thresholds:
            sp = SinglePass(t, all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
            s1 = normalized_mutual_info_score(label, sp.cluster_result, average_method='arithmetic')
            s1s.append(s1)
        index = s1s.index(max(s1s))
        sp = SinglePass(thresholds[index], all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = thresholds[index]
    return sp.cluster_result,threshold 
        
def evaluate(all_embeddings,label,block_num,pred_label,result_path,task):
    if task == "DRL":
        y_pred,threshold = DRL_cluster(all_embeddings,block_num,pred_label)
    elif task == "random":
        y_pred,threshold = random_cluster(all_embeddings,block_num,pred_label)
    elif task == "semi-supervised":
        y_pred,threshold = semi_cluster(all_embeddings,label,block_num,pred_label)
    elif task == "traditional":
        y_pred,threshold = NMI_cluster(all_embeddings,label,block_num,pred_label)
    
    #NMI
    s1 = normalized_mutual_info_score(label, y_pred, average_method='arithmetic')
    #AMI
    s2 = adjusted_mutual_info_score(label, y_pred, average_method='arithmetic')
    #ARI
    s3 = adjusted_rand_score(label, y_pred)
    
    print('** Theta:{:.2f} **\n'.format(threshold))
    print('** NMI: {:.2f} **\n'.format(s1))
    print('** AMI: {:.2f} **\n'.format(s2))
    print('** ARI: {:.2f} **\n'.format(s3))
    result = '\nmessage_block_'+str(block_num)+'\nthreshold: {:.2f} '.format(threshold)+'\n** NMI: {:.2f} **\n'.format(s1) + '** AMI: {:.2f} **\n'.format(s2) + '** ARI: {:.2f} **\n'.format(s3)

    if not os.path.exists(result_path) :
        pass
    else:
        with open(result_path,encoding='utf-8') as file:
            content=file.read()
        result = content.rstrip() + result
    file = open(result_path, mode='w')
    file.write(result)
    file.close()
    return y_pred