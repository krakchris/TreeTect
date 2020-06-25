FROM ubuntu
WORKDIR /
RUN apt update -y && apt install python3 -y && apt install python3-pip -y &&  pip3 install boto
ADD TreeTect/ms_point_data_comparison /
RUN pip3 install -r requirement.txt
RUN pip3 install awscli
WORKDIR src
ENTRYPOINT python3 point_data_comparison_wrapper.py; exit 0