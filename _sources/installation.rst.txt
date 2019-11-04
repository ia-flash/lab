Installation
============

Il est nécessaire de disposer d’une machine avec des GPU et un processeur
puissant, complété par un stockage rapide et sécurisé.  Actuellement, un
serveur avec 128 Gb de RAM, 4 disques SSD de 4Tb, 4 cartes graphiques Titan V
permet l’entraînement en un temps raisonnable. Ubuntu 16 serveur est installé,
la version des pilotes GPU est la 410.78, sous CUDA 10.0.  Les disques SSD sont
au format standard Ext4.

Dependencies
------------

Here are some examples to get you started.

.. code:: sh

  sudo apt update && \
  sudo apt upgrade && \
  sudo apt install gcc make dkms


Install nvidia drivers
----------------------

- Download drivers. Drivers version may be different depedending on GPU. Verify the driver number at [nvidia.com](https://www.nvidia.com/Download/index.aspx?lang=en-us)

.. code:: sh

  wget http://us.download.nvidia.com/XFree86/Linux-x86_64/430.50/NVIDIA-Linux-x86_64-430.50.run


Install drivers
---------------


.. code:: sh

  sudo sh ./NVIDIA-Linux-x86_64-430.50.run  --no-drm --disable-nouveau --dkms --silent --install-libglvnd

Install nvidia-docker
---------------------

.. code:: sh

  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker

- Install [nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime)

.. code:: sh

  sudo apt-get install nvidia-container-runtime

- (Optional) In order to use docker-compose with nvidia it's necessary to add the following settings at `/etc/docker/daemon.json`:

.. code:: json

  {
      "runtimes": {
          "nvidia": {
              "path": "nvidia-container-runtime",
              "runtimeArgs": []
          }
      }
  }

References:
    - `nvidia docs <https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html>`_
    - `github issue docker compose nvidia <https://github.com/docker/compose/issues/6691>`_



Install docker
--------------

.. code:: bash

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

Install docker-compose
----------------------

.. code:: bash

  sudo curl -L "https://github.com/docker/compose/releases/download/1.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

Install Lab IA Flash
--------------------

.. code:: bash

  git clone https://github.com/ia-flash/lab.git lab
  cd lab
  make up
