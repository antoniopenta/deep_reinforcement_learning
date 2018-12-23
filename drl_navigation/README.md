

#  Navigation Project

![](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

- More info can be read here : [Udacity Project link](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)


## Main Description

 - Deep-Q Learning Algorithm to solve the Banana game in Unity.
 - Jupyter notebook (in the jupyter folder) contains a description of the code
 - all the code is also saved here : https://github.com/antoniopenta/deep_reinforcement_learning/tree/master/drl_navigation




## Prerequisites

- create a python env as described here: https://github.com/udacity/deep-reinforcement-learning/tree/master/python

- or

```sh
$ pip install -r requirements.txt
```


- You need to download the unity env Banana at the following link ( 4 Mac) and save (unzipped) in the env folder of the project:  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip


## Project Explanation

- The report of the project in in the jupyter folder : notebook_navigation.ipynb


## Results (Final Model)

- The final model is stored in  the model folder : checkpoint_3.pth
- The score results are stored in the data folder : score_3.txt


## Train and Test Scripts


- To train the agent (activate your virtual env before) :
```sh
$ python main_script_dq_train.py
```

- To see the agent acting (activate your virtual env before) :
```sh
$ python main_script_dq_test.py
```





## Authors
- Antonio Penta