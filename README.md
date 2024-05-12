



## Environment setup
Use the following commands to setup the environment in your local system.

```
conda create -n cs276f python=3.10.12
conda activate cs276f
pip install --upgrade mani_skill
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stable-baselines3[extra]
pip install tyro wandb chardet
```
To test the setup run the following command.
```
python -m mani_skill.examples.demo_random_action
```


Download the demos 

```
!python -m mani_skill.utils.download_demo "StackCube-v1" -o "demos"
!python -m mani_skill.utils.download_demo "PickCube-v1" -o "demos"
```


Change the Wandb entity name in config.py