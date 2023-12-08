# HCRC

This repository contains the source code for the paper accepted by Knowledge-Based System *Unsupervised social event detection via hybrid graph contrastive learning and reinforced incremental clustering*.
The dataset is available from Zenodo(https://zenodo.org/records/7845451).

## Overview
Detecting events from social media data streams is gradually attracting researchers. The innate challenge for detecting events is to extract discriminative information from social media data thereby assigning the data into different events. Due to the excessive diversity and high updating frequency of social data, using supervised approaches to detect events from social messages is hardly achieved. To this end, recent works explore learning discriminative information from social messages by leveraging graph contrastive learning (GCL) and embedding clustering in an unsupervised manner. However, two intrinsic issues exist in benchmark methods: conventional GCL can only roughly explore partial attributes, thereby insufficiently learning the discriminative information of social messages; for benchmark methods, the learned embeddings are clustered in the latent space by taking advantage of certain specific prior knowledge, which conflicts with the principle of unsupervised learning paradigm. In this paper, we propose a novel unsupervised social media event detection method via hybrid graph contrastive learning and reinforced incremental clustering (HCRC), which uses hybrid graph contrastive learning to comprehensively learn semantic and structural discriminative information from social messages and reinforced incremental clustering to perform efficient clustering in a solidly unsupervised manner. We conduct comprehensive experiments to evaluate HCRC on the Twitter and Maven datasets. The experimental results demonstrate that our approach yields consistent significant performance boosts. In traditional incremental setting, semi-supervised incremental setting and solidly unsupervised setting, the model performance has achieved maximum improvements of 53%, 45%, and 37%, respectively.

## Requirements

- Python version: 3.7.10
- Pytorch version: 1.8.1 + cu101
- torch-geometric version: 2.0.2

## How to Run

You can run the file with above mentioned hyperparameters
```
python main.py --task TASK --result_path PATH
```
`--task:`
Name of the task. Supported names are: DRL, random, semi-supervised, traditional. Default is DRL.  
usage example :`--task DRL`

`--result_path:`
Path for saving experimental results. Default is res.txt. 
usage example :`--result_path res.txt`
