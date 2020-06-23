FROM ubuntu
WORKDIR /
RUN apt update -y && apt install python3 -y && apt install python3-pip -y &&  pip3 install boto
ADD /ms_trainning/TreeTect/utils/ /root
ADD /ms_trainning/TreeTect/training_data_generation/ /root
WORKDIR training_data_generation
RUN pip3 install -r requirements.txt
RUN pip3 install awscli
WORKDIR src
ENTRYPOINT python3 training_data_generation_wrapper.py; exit 0