FROM ubuntu
WORKDIR /
RUN apt update -y && apt install python3 -y && apt install python3-pip -y &&  pip3 install boto
ADD ./TreeTect/ms_file_chunker/ /
RUN pip3 install -r requirements.txt
RUN pip3 install awscli
WORKDIR src
ENTRYPOINT python3 file_chunker_wrapper.py; exit 0
