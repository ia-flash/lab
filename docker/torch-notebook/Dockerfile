FROM mmdetection
RUN conda config --set auto_update_conda false
RUN conda install jupyter

RUN jupyter notebook --generate-config
#RUN conda create -n env_name pip
#RUN source activate env_name;conda install pip
#RUN pip install dataiku-api-client
RUN git clone https://github.com/dataiku/dataiku-api-client-python.git

WORKDIR dataiku-api-client-python
RUN cat setup.py
RUN /opt/conda/bin/python setup.py install

RUN pip install pandas vertica-python
RUN pip install flask
WORKDIR /workspace
# to launch the service:
#jupyter notebook --ip=0.0.0.0 --no-browser --allow-root


#FROM jupyter/minimal-notebook
# USER root
# RUN useradd -ms /bin/bash journevi
# USER journevi
# WORKDIR /home/journevi

