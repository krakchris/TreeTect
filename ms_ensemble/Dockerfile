FROM ubuntu:latest
FROM jimgroot/treetech:tensorflowmodules
WORKDIR /
RUN apt-get update
ADD ./TreeTect/ms_ensemble ms_ensemble/
ADD ./TreeTect/utils utils/ 
RUN pip3 install -r ms_ensemble/requirement.txt
RUN pip3 install awscli
WORKDIR /ms_ensemble/src
ENTRYPOINT python3 ensemble_wrapper.py; exit 0
