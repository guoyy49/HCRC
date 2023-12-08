import torch
import gc
import torch.nn.functional as F
from models import Graph_ModelTrainer,Node_ModelTrainer
gc.collect()
torch.cuda.empty_cache()
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import utils
from evaluate import evaluate

def main():
    args, unknown = utils.parse_args()
   
    for i in range(22):
        print("************Message Block "+str(i)+" start! ************")
        #Node-level learning
        embedder_N = Node_ModelTrainer(args,i)
        Node_emb,label = embedder_N.get_embedding()
        #Graph-level learning
        embedder_G = Graph_ModelTrainer(args,i)
        Graph_emb,label = embedder_G.get_embedding()
        #combining vectors
        if i==0:
            all_embeddings = np.concatenate((Graph_emb,Node_emb),axis=1)
            all_label = label
        else:
            temp = np.concatenate((Graph_emb,Node_emb),axis=1)
            all_embeddings = np.concatenate((all_embeddings,temp),axis=0)
            all_label = all_label+label
        all_embeddings = torch.tensor(all_embeddings)
        all_embeddings = F.normalize(all_embeddings, dim=-1, p=2).detach().cpu().numpy()
            
        #evaluate 
        if i == 0:
            pred_y = evaluate(all_embeddings,label,i,None,args.result_path,args.task)
            all_pred_y = pred_y
        else:
            pred_y = evaluate(all_embeddings,label,i,all_pred_y,args.result_path,args.task)
            all_pred_y = all_pred_y + pred_y
        print("************Message Block "+str(i)+" end! ************\n\n")
    
if __name__ == "__main__":
    main()