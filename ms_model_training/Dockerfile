FROM ubuntu:latest
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04
WORKDIR /
RUN apt-get update
RUN pip3 --version
ADD ./tensorflow /
ADD featute_msservice/TreeTect/ms_model_training ms_model_training/
ADD featute_msservice/TreeTect/utils utils/ 
RUN ls
RUN pip3 install -r ms_model_training/requirement.txt
RUN pip3 install awscli
RUN apt install git -y
WORKDIR TensorFlow/models/research/
RUN pip3 install .
RUN python3 setup.py build
RUN python3 setup.py install
WORKDIR slim
RUN python3 setup.py build
RUN python3 setup.py install
WORKDIR /
RUN git clone https://github.com/cocodataset/cocoapi
WORKDIR cocoapi/PythonAPI
RUN python3 setup.py build_ext install
WORKDIR /ms_model_training/src
RUN apt install liblzma-dev
ENTRYPOINT python3 training_wrapper.py; exit 0