# coding:utf-8

import numpy as np
import os
import torch.nn.functional as F
import torch
import copy
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import time
from easy_drl.utils import make_transition


class SinglePass:
    def __init__(self, sim_threshold, data, flag, label, size, agent, para, sim_init, sim=False, global_step=0):
        self.device = torch.device('cuda:0')
        self.text_vec = None  #
        self.topic_serial = None
        self.topic_cnt = 0
        self.sim_threshold = sim_threshold
        
        self.done_data = data[0:data.shape[0] - size]
        self.new_data = data[data.shape[0] - size:]
        self.done_label = label
        
        if flag == 0 or flag == 2:
            self.cluster_result = self.run_cluster(flag, size)
        else:
            self.agent = agent
            self.scheme = ["state", "action", "reward", "done", "log_prob"]
            self.global_step = global_step
            self.sim = sim
            if self.sim:
                start_time = time.time()
                self.cluster_result = self.run_cluster_sim(flag, size, para, sim_init, sim, data)  
                end_time = time.time()
                self.time = end_time - start_time
                print("Creating Environment Done! " + "It takes "+str(int(self.time))+" seconds.")
            else:
                start_time = time.time()
                self.pseudo_labels = self.run_cluster_init(0.6, size)
                if flag == 1:
                    self.text_vec = self.done_data
                    self.topic_serial = copy.deepcopy(self.done_label)
                    self.topic_cnt = max(self.topic_serial)
                state = self.get_state(sim, sim_init, data)
                action, action_log_prob = self.agent.select_action(state)
                # action projection
                sim_threshold = torch.clamp(action, -1, 1).detach()
                sim_threshold += 7
                sim_threshold /=10
                self.sim_threshold = sim_threshold.item()
                end_time = time.time()
                self.time = end_time - start_time
                print("Getting Threshold Done! " + "It takes "+str(int(self.time))+" seconds. ")
                print("Threshold is "+str(self.sim_threshold)+".\n")
                print("Evaluating message block...")
                start_time = time.time()
                self.cluster_result = self.run_cluster(flag, size)  # clustering
                end_time = time.time()
                self.time = end_time - start_time
                print("Done! " + "It takes "+str(int(self.time))+" seconds.\n")

    def clustering(self, sen_vec):
        if self.topic_cnt == 0:
            self.text_vec = sen_vec
            self.topic_cnt += 1
            self.topic_serial = [self.topic_cnt]
        else:
            sim_vec = np.dot(sen_vec, self.text_vec.T)
            max_value = np.max(sim_vec)

            topic_ser = self.topic_serial[np.argmax(sim_vec)]
            self.text_vec = np.vstack([self.text_vec, sen_vec])

            if max_value >= self.sim_threshold:
                self.topic_serial.append(topic_ser)
            else:
                self.topic_cnt += 1
                self.topic_serial.append(self.topic_cnt)
    
    def clustering_init(self, t, sen_vec):
        if self.topic_cnt_init == 0:
            self.text_vec_init = sen_vec
            self.topic_cnt_init += 1
            self.topic_serial_init = [self.topic_cnt_init]
        else:
            sim_vec = np.dot(sen_vec, self.text_vec_init.T)
            max_value = np.max(sim_vec)

            topic_ser = self.topic_serial_init[np.argmax(sim_vec)]
            self.text_vec_init = np.vstack([self.text_vec_init, sen_vec])

            if max_value >= t:
                self.topic_serial_init.append(topic_ser)
            else:
                self.topic_cnt_init += 1
                self.topic_serial_init.append(self.topic_cnt_init)
    
    def run_cluster_init(self, t, size):
        self.text_vec_init = []
        self.topic_serial_init = []
        self.topic_cnt_init = 0
        for vec in self.new_data:
            self.clustering_init(t,vec)
        return self.topic_serial_init
    
    def run_cluster_sim(self, flag, size, para, sim_init, sim, data):
        self.text_vec = []
        self.topic_serial = []
        self.topic_cnt = 0
        if flag == 1:
            self.text_vec = self.done_data
            self.topic_serial = copy.deepcopy(self.done_label)
            self.topic_cnt = max(self.topic_serial)
        for i, vec in enumerate(self.new_data):
            self.global_step += 1
            if i > 200:
                break
            if i > self.new_data.shape[0] * para:
                break
            state = self.get_state(sim, sim_init, data)
            
            action, action_log_prob = self.agent.select_action(state)
            self.sim_threshold = action.item()
            self.clustering(vec)
            
            reward = self.get_reward(sim_init, data)
            done = False
            transition = make_transition(self.scheme, state, action, reward, done, action_log_prob)
            self.agent.add_buffer(transition)
            if self.global_step % 200==0:
                self.agent.learn()

        return self.topic_serial[len(self.topic_serial) - size:]

    def run_cluster(self, flag, size):
        self.text_vec = []
        self.topic_serial = []
        self.topic_cnt = 0
        if flag == 1 or flag == 2:
            self.text_vec = self.done_data
            self.topic_serial = copy.deepcopy(self.done_label)
            self.topic_cnt = max(self.topic_serial)
        for i, vec in enumerate(self.new_data):
            self.clustering(vec)
        return self.topic_serial[len(self.topic_serial) - size:]

    def get_center(self,label,data):
        centers = []
        indexs_per_cluster = []
        label_u = list(set(label))
        for i in range(len(label_u)):
            indexs = [False] * data.shape[0]
            tmp_indexs_text = []
            for j in range(len(indexs)):
                if label[j] == label_u[i]:
                    indexs[j] = True
                    tmp_indexs_text.append(j)
            center = np.mean(data[indexs], 0).tolist()
            centers.append(center)
            indexs_per_cluster.append(tmp_indexs_text)
        return centers,indexs_per_cluster

    def get_info_cluster(self,text_vec,indexs_per_cluster):  # Get detailed clustering results
        res = []
        for i in range(len(indexs_per_cluster)):
            tmp_vec = []
            for j in range(len(indexs_per_cluster[i])):
                tmp_vec.append(text_vec[indexs_per_cluster[i][j]])
            tmp_vec = np.array(tmp_vec)
            res.append(tmp_vec)
        return res

    def get_state(self, sim, sim_init, data):  # get state of RL
        state = []
        if sim:
            data = data[sim_init:len(self.topic_serial)]
            topic_serial = self.topic_serial[sim_init:]
        else:
            data = self.new_data
            topic_serial = self.pseudo_labels
        centers,indexs_per_cluster = self.get_center(topic_serial, data)
        
        centers = np.array(centers)
        neighbor_dists = np.dot(centers, centers.T)
        
        neighbor_dists = np.nan_to_num(neighbor_dists, 0.0001)
        # the minimum neighbor distance
        state.append(neighbor_dists.min())
        # the average separation distance
        state.append((neighbor_dists.mean() * max(topic_serial) - 1) / max(topic_serial))
        info_of_cluster = self.get_info_cluster(data,indexs_per_cluster)

        coh_dists = 0
        for cluster in info_of_cluster:
            if cluster.shape[0] == 1:
                continue
            else:
                sums = cluster.shape[0] * (cluster.shape[0] - 1) / 2
            tmp_vec = np.array(cluster)
            cohdist = np.dot(tmp_vec, tmp_vec.T)
            if cohdist.max() > coh_dists:
                coh_dists = cohdist.max()
#         Dunn index
        state.append(neighbor_dists.min()/coh_dists)
    
        #Sum of intra-group error squares
        SSE = 0
        SSEE = 0
        for i in range(len(indexs_per_cluster)):
            sumtmp = 0
            for j in range(len(indexs_per_cluster[i])):
                tmp = np.dot(data[indexs_per_cluster[i][j]].T,centers[i])
                SSE = SSE + (tmp)**2
                sumtmp = sumtmp + (tmp)**2
            SSEE = SSEE + sumtmp/len(indexs_per_cluster[i])
#         state.append(SSE)

        # Sum of squared errors between groups
        SSR = 0
        SSRR = 0
        for i in range(len(centers)):
            SSR = SSR + np.dot(centers[i].T,centers.mean(axis=0))
            SSRR = SSRR + np.dot(centers[i].T,centers.mean(axis=0))**2
        SSRR = SSRR / max(topic_serial)
#         state.append(SSR)
        #the average cohesion distance
        coh_dists = 0
        for cluster in info_of_cluster:
            if cluster.shape[0] == 1:
                continue
            else:
                sums = cluster.shape[0] * (cluster.shape[0] - 1) / 2
            tmp_vec = np.array(cluster)
            cohdist = np.dot(tmp_vec, tmp_vec.T)
            cohdist = np.maximum(cohdist, -cohdist)
            coh_dists = coh_dists + (cohdist.sum() - cluster.shape[0]) / (2 * sums + 0.0001)
        state.append(coh_dists / max(topic_serial))

        state.append(silhouette_score(data, topic_serial, metric='euclidean'))
        return np.array(state)

    def get_reward(self, sim_init, data):  # get reward of RL

        data = data[sim_init:len(self.topic_serial)]
        topic_serial = self.topic_serial[sim_init:]
        
        return calinski_harabasz_score(data, topic_serial)
