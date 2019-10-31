# Lab IA Flash

Training, inference, image vizualisation, database connector scripts.

## Installation

### Dependencies

```bash
sudo apt update && \
sudo apt upgrade && \
sudo apt install gcc make dkms
```

### Install nvidia drivers

* Download drivers. Drivers version may be different depedending on GPU. Verify the driver number at [nvidia.com](https://www.nvidia.com/Download/index.aspx?lang=en-us)

```bash
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/430.50/NVIDIA-Linux-x86_64-430.50.run
```

* Install drivers

```bash
sudo sh ./NVIDIA-Linux-x86_64-430.50.run  --no-drm --disable-nouveau --dkms --silent --install-libglvnd
```

* Install nvidia-docker

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

* Install [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime)

```bash
sudo apt-get install nvidia-container-runtime
```

* (Optional) In order to use docker-compose with nvidia it's necessary to add the following settings at `/etc/docker/daemon.json`:


```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

References:
* [nvidia docs](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html)
* [github issue docker compose nvidia](https://github.com/docker/compose/issues/6691)

### Install docker

```bash
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo usermod -aG docker $USER
```

### Install docker-compose

```
sudo curl -L "https://github.com/docker/compose/releases/download/1.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

### Install Lab IA Flash

```
git clone https://github.com/ia-flash/lab.git lab
cd lab
make up
```

## Usage (FR)

### Vehicule detection

Nous avons mis à disposition le code pour la détection sur github. Il sont sur https://github.com/ia-flash/lab
Dans le répertoire de détection, on peut lancer la commande : 

```
cd iaflash/detection
cfg=/workspace/mmdetection/configs/retinanet_r50_fpn_1x.py
model=/model/retina/retinanet_r50_fpn_1x_20181125-7b0c2548.pth
gpu=4
```

### Lancement de la detection

```
./dist_test.sh $cfg $model $gpu
```

Cela lancera la détection à partir des images de `files_trunc`. Les résultats
seront consignés dans la table `box_detection`. On peut changer ces tables dans
le script test.py. Le shell `dist_test.sh` parallélise proprement `test.py` sur les
GPU, sans avoir à gérer les process fils dans un script python. 

La connection à la base de données doit être spécifiée dans le fichier de
configuration `/docker/env.list`. Cette connection est transparente vis-à-vis du
connecteur : postgres ou vertica. La dépendance à DSS est évitée, puisqu’on se
connecte directement au gestionnaire de DB. Il faut seulement (avec DSS par
exemple) créer le shéma de la table de sortie `box_detection`.
