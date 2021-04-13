FROM balenalib/raspberrypi3-debian:buster

ARG DEBIAN_FRONTEND=noninteractive
ENV READTHEDOCS True
ENV NODE_OPTIONS "--max-old-space-size=2048"


#do installation
RUN apt-get update \
	&& apt-get install -y --no-install-recommends openssh-server \
    && apt-get install -y libopencv-dev python-opencv libopenjp2-7 libxcb1 libatlas-base-dev \
    && apt-get install -y tzdata git python3-pip python3-numpy wget \
    && apt-get autoremove \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/* \
#do users
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && mkdir /var/run/sshd
	
	
RUN export DEBIAN_FRONTEND=noninteractive \
    && ln -fs /usr/share/zoneinfo/Asia/Taiwan /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py --user
RUN python3 -m pip install numpy


#nodejs for notebooks
# RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
#     && apt-get install -y nodejs

#python libraries
RUN pip3 install -U pip \
    && python3 -m pip install --no-cache-dir -U grpcio absl-py py-cpuinfo psutil portpicker grpcio six mock requests gast astor termcolor \
    && python3 -m pip install grpcio grpcio-tools \
    && python3 -m pip install  --upgrade --force-reinstall pyzmq \
    && python3 -m pip install supervisor tornado python-periphery \
    && python3 -m pip install google-auth oauthlib imutils matplotlib


#downloading library file 
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - \
    && apt-get update \
    && apt-get install -y python3-edgetpu
    # && wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl \
    # && wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.3.0/tensorflow-2.3.0-cp37-none-linux_armv7l.whl

# install tensorflow to run facenet
RUN pip install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.3.0/tensorflow-2.3.0-cp37-none-linux_armv7l.whl tqdm \
    && pip3 install opencv-python
# RUN rm /tensorflow-2.3.0-cp37-none-linux_armv7l.whl /tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl


#copy files

RUN git clone https://github.com/yaotechiu/tflite --depth 1 \
    && git clone https://github.com/Kao1126/EdgeTPU-FaceNet \
    && /tflite/python/examples/detection/install_requirements.sh
#loading pretrained models

# RUN ls /tflite/python/examples/detection/





#copy supervisord files
COPY "./conf/supervisord.conf" /etc/supervisor/conf.d/supervisord.conf
RUN mkdir /var/log/supervisord/

#copy FaceNet GRPC


RUN mkdir /root/Test/
RUN mkdir /root/Test/GRPC/
RUN mkdir /root/Test/copy/


COPY "./Facenet/" /root/Test/GRPC/Facenet

#supervisord
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

#opencv2 bug on arm (Undefined reference to __atomic)
ENV LD_PRELOAD "/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0"

#SSH Port
EXPOSE 22 8888 8080

#set stop signal
STOPSIGNAL SIGTERM

WORKDIR /root/Test/GRPC/Facenet

RUN ls





