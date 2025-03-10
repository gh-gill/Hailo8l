#### Info on WSL commands
```bash
#Install Windows Subsystem for Linux.
wsl.exe --install
#check status
wsl--status
#list all installed linux distros
wsl --list --all
#list running linux distros
wsl --list --running
#get online available linux distros
wsl --list --online
#install a distro
wsl --install Ubuntu-22.04
#Change wsl version number 1 or 2 on distro
wsl –set-version [distro name] [wsl version number] 
#open a specific linux distro
wsl -d <Distro Name>
#shutdown linux enviroment
wsl --shutdown
```

# How to Setup Raspberry Pi 5 with Hailo8l AI Kit using yolov8s on Windows (WSL2 Ubuntu-22.02) Custom Objects and Labels

### On Windows WSL Ubuntu setup (If using docker skip this step)
```bash
wsl --install Ubuntu-22.04
cd ~
mkdir data
``` 

### Alternative with Docker
```bash
docker pull ubuntu:jammy
```

##### Create a Persisting Data folder "~/Documents/docker/hailo_yolov8s"
```bash
docker run -it --name hailo_yolov8s -v ~/Documents/docker/hailo_yolov8s:/data ubuntu:jammy /bin/bash
```

#### Create container
```bash
cd data
```

```bash
apt-get update
apt-get install git
apt-get install software-properties-common
```

### Get Guide
```bash
git clone https://github.com/gh-gill/Hailo8l.git
```

### Add deadsnakes repository
#### Note if using Docker use all commands without sudo
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

### Training
#### Setup virtual env, activate and install packages
```bash
cd Hailo8l
sudo apt-get install libpython3.11-stdlib libgl1-mesa-glx
sudo apt install python3.11 python3.11-venv
python3.11 -m venv venv_yolov8
source venv_yolov8/bin/activate
pip install ultralytics
```
#### Create Dataset from Data created by Label Studio
###### --datapath = "data" folder from Label Studio
###### copy downloaded dataset folder from LabelStudio to ~/datasets/
###### this will create the folder structure
###### ~/Hailo8l/datasets/images/train */val 
###### ~/Hailo8l/datasets/labels/train */val                                        
```bash
python steps/2_install_dataset/train_val_split.py --datapath="/data/Hailo8l/datasets/data" --train_pct=.8
```

##### Edit files
###### Edit "/data/Hailo8l/config.yaml", "/data/Hailo8l/labels.json", "/data/Hailo8l/config/postprocess_config/yolov8s_nms_config.json" & "/data/Hailo8l/steps/4_test/labels.txt" to match your dataset

##### train model as pytorch
```bash
yolo detect train data=/data/Hailo8l/config.yaml model=yolov8s.pt name=retrain_yolov8s project=./model/runs/detect epochs=100 batch=16
```

### Convert to ONNX
```bash
cd /data/Hailo8l/model/runs/detect/retrain_yolov8s/weights   
```

```bash
yolo export model=./best.pt imgsz=640 format=onnx opset=11 
```

```bash
cd /data/Hailo8l && deactivate
```

### Install Hailo
```bash
sudo apt-get install python3.8 python3.8-venv python3.8-dev
```

```bash
python3.8 -m venv venv_hailo
source venv_hailo/bin/activate
sudo apt-get update
sudo apt-get install build-essential python3-dev graphviz graphviz-dev python3-tk
pip install pygraphviz
```
#### Download files from https://hailo.ai/developer-zone/software-downloads/ and place in "hmoe/hmi/Hailo8l/whl/"

```bash
pip install whl/hailo_dataflow_compiler-3.27.0-py3-none-linux_x86_64.whl
```

```bash
pip install whl/hailo_model_zoo-2.11.0-py3-none-any.whl
```

```bash
git clone https://github.com/gh-gill/hailo_model_zoo.git
```

### Install Custom dataset
```bash
python steps/2_install_dataset/create_custom_tfrecord.py val
```
```bash
python steps/2_install_dataset/create_custom_tfrecord.py train
```
### Parse
```bash
python steps/3_process/parse.py
```
### Optimize
```bash
python steps/3_process/optimize.py
```
### Compile
```bash
python steps/3_process/compile.py
```

## Raspbery Pi 5

```bash
cd hailo8l
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
pip install setproctitle

```

```bash
cd hailo-rpi5-examples
source setup_env.sh
cd ..
```

### Run Object Detection
```bash
python hailo-rpi5-examples/basic_pipelines/detection.py -i rpi --hef best_quantized_model.hef --labels-json labels.json
```


# How to Setup Raspberry Pi 5 with Hailo8l AI Kit using yolov11n on Windows (WSL2 Ubuntu 24.04) Custom Objects and Labels

### On Windows WSL Ubuntu setup (If using docker skip this step)
```bash
wsl --install Ubuntu-22.04
``` 

### Get Guide
```bash
git clone https://github.com/BetaUtopia/Hailo8l.git
```

### Add deadsnakes repository
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

### Training
```bash
cd Hailo8l
sudo apt-get update
sudo apt-get install libpython3.12-stdlib libgl1-mesa-glx
sudo apt install python3.12 python3.12-venv
python3.12 -m venv venv_yolov11
source venv_yolov11/bin/activate
pip install ultralytics
```
#### Create Dataset from Data created by Label Studio
###### --datapath = "data" folder from Label Studio
###### copy downloaded dataset folder from LabelStudio to ~/datasets/
###### this will create the folder structure
###### ~/Hailo8l/datasets/images/train */val 
###### ~/Hailo8l/datasets/labels/train */val                                        
```bash
python steps/2_install_dataset/train_val_split.py --datapath="/home/hmi/Hailo8l/datasets/data" --train_pct=.8
```
##### Edit files
###### Edit "config.yaml" & "labels.json" to match your dataset

##### train model as pytorch
```bash
yolo detect train data=/home/hmi/Hailo8l/config_yolov11s.yaml model=yolo11s.pt name=retrain_yolov11n project=./model/runs/detect epochs=100 batch=16
```
### Convert to ONNX
```bash
cd /home/hmi/Hailo8l/model/runs/detect/retrain_yolov11s/weights   
```

```bash
yolo export model=./best.pt imgsz=640 format=onnx opset=11 
```

```bash
cd ~/Hailo8l && deactivate
```

### Install Hailo
```bash
sudo apt-get install python3.10 python3.10-venv python3.10-dev
```

```bash
python3.10 -m venv venv_hailo
source venv_hailo/bin/activate
sudo apt-get update
sudo apt-get install build-essential python3-dev graphviz graphviz-dev python3-tk
pip install pygraphviz scipy==1.9.3
```

#### Download files from https://hailo.ai/developer-zone/software-downloads/ and place in "hmoe/hmi/Hailo8l/whl/"

```bash
pip install whl/hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl
```

```bash
pip install whl/hailo_model_zoo-2.14.0-py3-none-any.whl
```

Parse
```bash
python steps/3_process/parse_yolo11s.py
```
Optimize
```bash
python steps/3_process/optimize_yolo11s.py
```
Compile
```bash
python steps/3_process/compile_yolo11s.py
```

### Raspbery Pi

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install hailo-all
sudo reboot
```

Check Versions

```bash
hailortcli fw-control identify
```

```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
```

```bash
./install.sh
```

```bash
source setup_env.sh
```

```bash
python basic_pipelines/detection.py -i rpi --hef /home/pi/Hailo8l/best.hef --labels-json /home/pi/Hailo8l/labels_yolov11n.json
```

### Raspbery Pi with picamera2

```bash
pip uninstall opencv-python --break-system-packages
pip install opencv-python-headless --break-system-packages
```

```bash
python steps/4_test/detect_picamera2.py
```
