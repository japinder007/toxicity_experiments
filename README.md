# Detecting-Cyberbullying-Across-SMPs

This is based on - https://github.com/sweta20/Detecting-Cyberbullying-Across-SMPs.

Abstract. Harassment by cyberbullies is a significant phenomenon on the social media. Existing works for cyberbullying detection have at least one of the following three bottlenecks. First, they target only one particular social media platform (SMP). Second, they address just one topic of cyberbullying. Third, they rely on carefully handcrafted features of the data. We show that deep learning based models can overcome all three bottlenecks. Knowledge learned by these models on one dataset can be transferred to other datasets. We performed extensive experiments using three real-world datasets: Formspring (˜12k posts), Twitter (˜16k posts), and Wikipedia(˜100k posts). Our experiments provide several useful insights about cyberbullying detection. To the best of our knowledge, this is the first work that systematically analyzes cyberbullying detection on various topics across multiple SMPs using deep learning based models and transfer learning.

## Dataset

The three datasets used in the paper can be downloaded from [here](https://drive.google.com/open?id=11RMLCSIAO3dWk9ejSkVYc5tQwwK5pquG).

Please download the dataset and unzip at data/.

We have also used two different kind of embeddings for initialization which can be found at the mentioned links.

- [Sentiment Specific word embeddings (SSWE)](http://ir.hit.edu.cn/~dytang/paper/sswe/embedding-results.zip)
- [GLoVe](https://nlp.stanford.edu/projects/glove/)


### Prerequisites

We will be using the tensorflow docker image. You can use the following command (update paths accordingly)

sudo docker run --runtime=nvidia -it --rm -p 9000:8888 -v /home/japinder:/notebooks tensorflow/tensorflow:latest-gpu

The above commamd will start a jupyter notebook and you can connect to it from your laptop. Before you do that, on your
laptop create a ssh-tunnel like so
ssh  -N -f -L 8158:dev-ml17:9000 dev-ml17

Then you can connect to localhost:8158 on your laptop.

There are several ports above
  - 8888: The port inside the container on which the notebooks server runs.
  - 9000: The port on the server (outside the container) which is mapped to port 8888 inside the container.
  - 8158: The port on your laptop. You should open localhost:8158 in a browser on your laptop and if everything is setup correctly, that maps to port 9000 on the server outside the container, which in turn is mapped to port 8888 inside the container.
