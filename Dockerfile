FROM debian:bullseye

# Install base utilities
RUN apt-get update && apt-get install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda

RUN /opt/conda/bin/pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

ADD monoprune /root/project/monoprune

RUN /opt/conda/bin/pip3 install -e /root/project/monoprune

ENV PATH="$PATH:/opt/conda/bin"

# FROM ubuntu:jammy
# 
# RUN apt-get update && apt-get install -y python3.10 python3-pip
# 
# RUN pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
# 
# ADD monoprune /root/monoprune
# 
# RUN pip3 install -e /root/monoprune