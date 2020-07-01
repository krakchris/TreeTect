FROM ubuntu
WORKDIR /
RUN apt update -y && apt install python3 -y && apt install python3-pip -y &&  pip3 install boto
ADD ./TreeTect/utils/ /utils
ADD ./TreeTect/ms_training_data_generation/ /ms_training_data_generation
RUN pip3 install -r ms_training_data_generation/requirements.txt
RUN pip3 install awscli
WORKDIR ms_training_data_generation/src
ENTRYPOINT python3 training_data_generation_wrapper.py; exit 0
