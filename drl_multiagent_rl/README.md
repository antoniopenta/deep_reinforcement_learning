

#  Tennis Project (Multi Agent Reinforcement Learning)

![](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

- More info about the purpose of the project can be read here : [Udacity Project link](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)


## Project Explanation

- The methodology (project execution) is described in the file Report.ipynb


## Prerequisites

- create a python env as described here: https://github.com/udacity/deep-reinforcement-learning/tree/master/python

- or

```sh
$ pip install -r requirements.txt
```







## Results (Final Model)

- The final model is stored in  the model folder :
    - first_checkpoint_agent_1_version_1.pth (first model with Average Score of 0.50),
    - first_checkpoint_agent_2_version_1.pth (first model with Average Score of 0.50),
    - best_checkpoint_agent_1_version_1.pth (best model with Average Score of 1.35),
    - best_checkpoint_agent_1_version_1.pth (best model with Average Score of 1.35),

- The score results are stored in the data folder : score_1.txt

- The parameters are stored in the json file : version_1.json


## Train and Test Scripts


- To train the agent (activate your virtual env before) :
```sh
$ python main_script_maddpg_train.py
```

- To see the agent acting (activate your virtual env before) :
```sh
$ python main_script_maddpg_test.py
```





## Authors
- Antonio Penta