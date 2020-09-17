FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
      apt-get install -y apt-utils && \
      apt-get install -y \
            wget \
            curl \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libpq-dev \
            python3 \
            python-opencv \
            python3-pip && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && \
  apt-get install -y git-all && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN git config --global  http.proxy $http_proxy
RUN git config --global  https.proxy $https_proxy

RUN pip install --upgrade pip

COPY . /workspace/iaflash
# mmcv requires cython==0.29.12 installed
RUN pip3 install -r /workspace/iaflash/docker/torch-notebook/requirements.txt

# install mmcv
WORKDIR /workspace
ENV CUDA_HOME=/usr/local/cuda
RUN git clone https://github.com/open-mmlab/mmcv.git --branch v1.1.1 mmcv
RUN cd /workspace/mmcv && \
    MMCV_WITH_OPS=1 pip install -e .

WORKDIR /workspace
RUN git clone https://github.com/open-mmlab/mmdetection.git --branch v2.4.0 mmdetection
RUN cd /workspace/mmdetection && \
    pip install -r requirements/build.txt && \
    pip install -v -e .

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
