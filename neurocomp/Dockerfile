# tested on Raspberry Pi 4 - Bullseye - 32-Bit
FROM balenalib/raspberrypi3:buster

# define build arguments
#ARG INSTALL_DIR=/opt/intel/openvino
#ARG INSTALL_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.2/linux/
#ARG INSTALL_FILE=l_openvino_toolkit_debian9_arm_2022.2.0.7713.af16ea1d79a_armhf.tgz
#ARG MODEL_DIR=/root/model
#ARG MODEL_FILE=https://raw.githubusercontent.com/heath-smith/neuromorphic-computing/main/neurocomp/test_inference_graph.xml
#ARG WEIGHTS_FILE=https://github.com/heath-smith/neuromorphic-computing/blob/main/neurocomp/test_inference_graph.bin?raw=true
#ARG TEST_IMAGE=https://github.com/heath-smith/neuromorphic-computing/blob/main/datasets/benchmark_test.png

# update apt-get
# install packages
# clean up resources
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    automake \
    cmake \
    cpio \
    gcc \
    g++ \
    libatlas-base-dev \
    libstdc++6 \
    libtool \
    libusb-1.0.0-dev \
    python3.7-dev \
    lsb-release \
    make \
    python3-pip \
    python3-numpy \
    python3-scipy \
    libgtk-3-0 \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    sudo \
    udev \
    unzip \
    vim \
    wget \
    git && \
    pip3 install opencv-python && \
    rm -rf /var/lib/apt/lists/*

ADD  BertModels /models

COPY setup.sh .

# make the installation directory
# download and unpack the deployment toolkit
# add usb rules
# download model files and test files


##     mkdir ${MODEL_DIR} && \
##     /bin/bash -c "source $INSTALL_DIR/setupvars.sh
##     cd ${INSTALL_DIR}/samples/c/ && \
##     sh build_samples.sh && \
##     cd ${INSTALL_DIR}/samples/cpp/ && \
##     sh build_samples.sh && \
##     wget --no-check-certificate ${MODEL_FILE} -O ${MODEL_DIR}/model.xml && \
##     wget --no-check-certificate ${WEIGHTS_FILE} -O ${MODEL_DIR}/model.bin && \
##     wget --no-check-certificate ${TEST_IMAGE} -O ${MODEL_DIR}/benchmark.png && \
##     cd /root/inference_engine_._samples_build/armv7l/Release/ && \
##     ./benchmark_app -m ${MODEL_DIR}/model.xml -i ${MODEL_DIR}/benchmark.png -d MYRIAD"

# COPY ./startup.sh /

# set the startup command for the container
#ENTRYPOINT [ "/startup.sh" ]
