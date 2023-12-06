FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# 1.8.0-cuda11.1-cudnn8-runtime
ARG user
ARG UID
ENV user $user
ENV UID $UID

ENV http_proxy="http://proxy1.zmnh.uni-hamburg.de:8888/" \
    https_proxy="http://proxy1.zmnh.uni-hamburg.de:8888/" \
    ftp_proxy="http://proxy1.zmnh.uni-hamburg.de:8888/" \
    rsync_proxy="http://proxy1.zmnh.uni-hamburg.de:8888/" \
    no_proxy="localhost,127.0.0.1,127.0.0.0/8,172.22.0.0/16,bonn-pe940.zmnh.uni-hamburg.de,bonn-dgx1.zmnh.uni-hamburg.de,intranet.zmnh.uni-hamburg.de,int.ims.bio,registry.ims.bio"

# --- Install any needed packages specified in requirements.apt
COPY . /
RUN apt-get update
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN xargs apt-get install -y --no-install-recommends < /requirements.apt
# Make python3 standard
RUN ln -s /usr/bin/python3.6 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip
# Install PIP
RUN pip install --upgrade pip
# --- Install any needed packages specified in requirements.pip
# RUN pip install -r /requirements.pip --upgrade
RUN pip install -r /requirements-long.pip
#RUN apt-get update && apt-get install libgl1
# Add a non-root user
#RUN useradd -rm -d /home/$user -s /bin/bash -g root -G sudo -u $UID $user
#USER $user
#WORKDIR /home/$user
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility








