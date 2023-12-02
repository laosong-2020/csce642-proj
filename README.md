# csce642-proj
## Set up Envs
### Set up Conda Virtual Env
```shell
conda create -n csce642-proj python=3.8
conda activate csce642-proj
```
### Install Sumo Latest Version
```shell
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
### Set SUMO Environment Variable
```shell
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
### Install Dependencies Packages
```shell
pip install -r requirements.txt
```
### Start Training
```shell
python main.py
```