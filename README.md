

### Overview
Detecting events from social media data streams is gradually attracting researchers. Due to the excessive diversity and high updating frequency of social data, using supervised approaches to detect events from social messages is hardly achieved. To this end, recent works explore learning discriminative information from social messages by leveraging graph contrastive learning and embedding clustering in an unsupervised manner. However, two intrinsic issues exist in benchmark methods: conventional graph contrastive learning can only roughly explore partial attributes, thereby insufficiently learning the semantic information of social messages; for benchmark methods, the learned embeddings are clustered in the latent space by taking advantage of certain specific prior knowledge, which conflicts with the principle of unsupervised learning paradigm. In this paper, we propose a novel social media event detection method, which uses hybrid graph contrastive learning to learn comprehensive semantic and structural information from social messages and reinforced incremental clustering to perform efficient online clustering in a solidly unsupervised manner. The experimental results show that our approach yields consistent significant performance boosts.

### Requirements

- Python version: 3.7.10
- Pytorch version: 1.8.1 + cu101
- torch-geometric version: 2.0.2

### How to Run

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
