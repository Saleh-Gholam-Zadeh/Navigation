# CrowdNav
This repository contains the implementation of my master thesis.



## Setup
 Installation is nothing more than installation of these githubs.

 a) https://github.com/vita-epfl/CrowdNav

 b) https://github.com/vita-epfl/trajnetplusplustools

 c) https://github.com/vita-epfl/trajnetplusplusbaselines

 d) https://github.com/vita-epfl/trajnetplusplusdataset

An alternative way is to download my anaconda virtual environment @(put address) and put it inside Anaconda/envs




## Getting started
If you want to use pretrained perception module (transfer-learning) use My code  in the master branch 






If you want to use pretrained perception module (transfer-learning) use My code in the master branch 

If you want to train




Below are the instructions for training and testing in the “others” branch



1. Train a policy.
```
python train.py --policy sarl
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy sarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python crowd_nav/utils/plot.py data/output/output.log
```


## Simulation Videos
CADRL             | LSTM-RL
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/vrWsxPM.gif" width="400" />|<img src="https://i.imgur.com/6gjT0nG.gif" width="400" />
SARL             |  OM-SARL
<img src="https://i.imgur.com/rUtAGVP.gif" width="400" />|<img src="https://i.imgur.com/UXhcvZL.gif" width="400" />


## Learning Curve
Learning curve comparison between different methods in an invisible setting.

<img src="https://i.imgur.com/l5UC3qa.png" width="600" />


