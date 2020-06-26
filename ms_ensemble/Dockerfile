FROM ubuntu:latest
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04
WORKDIR /
RUN apt-get update
ADD ./tensorflow /
ADD TreeTect/ms_ensemble ms_ensemble/
ADD TreeTect/utils utils/ 
RUN pip3 install -r ms_ensemble/requirement.txt
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
WORKDIR /ms_ensemble/src
ENTRYPOINT python3 ensemble_wrapper.py; exit 0