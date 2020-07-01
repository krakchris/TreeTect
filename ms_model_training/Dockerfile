FROM ubuntu:latest
FROM jimgroot/treetech:tensorflowmodules
WORKDIR /
RUN apt-get update
ADD ./TreeTect/ms_model_training ms_model_training/
ADD ./TreeTect/utils utils/ 
RUN pip3 install -r ms_model_training/requirement.txt
RUN pip3 install awscli
WORKDIR /
WORKDIR /ms_model_training/src
RUN apt install liblzma-dev
ENTRYPOINT python3 training_wrapper.py; exit 0
