



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

## To Run
Head to the main.ipynb and run each cell (if required change the name of the environment) to reproduce the results.
Note : It would take around 15hrs to reproduce results for each environment.