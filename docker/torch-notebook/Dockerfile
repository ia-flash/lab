FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && \
      apt-get install -y apt-utils && \
      apt-get install -y \
            wget \
            curl \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libpq-dev \
            libglib2.0-0 \
            ffmpeg \
            ninja-build \
            python3 \
            python-opencv \
            python3-pip && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && \
  apt-get install -y git-all && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN git config --global  http.proxy $http_proxy
RUN git config --global  https.proxy $https_proxy

# RUN pip install --upgrade pip

COPY . /workspace/iaflash
# mmcv requires cython==0.29.12 installed
RUN pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org pytest-xdist
RUN pip install -r /workspace/iaflash/docker/torch-notebook/requirements.txt --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org

# install mmcv
WORKDIR /workspace
RUN git clone https://github.com/open-mmlab/mmcv.git --branch v1.2.1 /mmcv
WORKDIR /mmcv
RUN MMCV_WITH_OPS=1;CUDA_HOME=/usr/local/cuda pip install -e .
#RUN MMCV_WITH_OPS=1;CUDA_HOME=/usr/local/cuda pip install mmcv-full
#RUN MMCV_WITH_OPS=1;CUDA_HOME=/usr/local/cuda pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git --branch v2.7.0 /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

RUN mkdir /.cache && chmod -R a+rw /.cache/

#END OF MMDET INSTALL
WORKDIR /workspace
RUN python3 -m pip install jupyter

RUN jupyter notebook --generate-config


#RUN git clone https://github.com/dataiku/dataiku-api-client-python.git
WORKDIR dataiku-api-client-python

# END OF JUPYTER AND VERTICA INSTALL
WORKDIR /workspace/iaflash
RUN  pip install -e .

CMD [ "python3", "./iaflash/app/app.py" ]
