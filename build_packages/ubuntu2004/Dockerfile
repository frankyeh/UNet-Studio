FROM rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.13.1 as builder-stage


ENV DEBIAN_FRONTEND noninteractive

# for CMAKE

RUN apt update && apt full-upgrade -y && \
  apt install -y --no-install-recommends \
  unzip \
  curl \
  git \
  ca-certificates \
  gcc \
  g++ \
  python3-pip \
  libglib2.0-0 \
  zlib1g-dev \
  mesa-utils \
  libgl1-mesa-dev \
  freeglut3-dev \
  libxkbcommon-x11-0 \
  libfontconfig1 \
  libfreetype6 \
  libdbus-1-3 && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#install cmake 
RUN curl -fsSL https://apt.kitware.com/kitware-archive.sh | bash
RUN apt install -y --no-install-recommends cmake ninja-build

#install Qt6
RUN pip install aqtinstall && aqt install-qt linux desktop 6.5.0 -m qtcharts --outputdir /opt/qt6
RUN /opt/qt6/6.5.0/gcc_64/bin/qmake --version

ENV QTDIR="/opt/qt6/6.5.0/gcc_64"
ENV PATH="$QTDIR/bin:$PATH"
ENV LD_LIBRARY_PATH="$QTDIR/lib:$LD_LIBRARY_PATH"


RUN cd /opt \
  && git clone https://github.com/frankyeh/UNet-Studio.git \
  && mv UNet-Studio src \
  && cd /opt/src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mkdir -p /opt/src/build

RUN cd /opt/src \
  && export TORCH_DIR="$(pip3 show torch | grep 'Location:' | awk '{print $2}')" \
  && cmake -S . -B build -GNinja -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_PREFIX_PATH=${TORCH_DIR}/torch -DCMAKE_BUILD_TYPE:STRING=Release -DTIPL_DIR=. \
  && cmake --build ./build --parallel --config Release

RUN mkdir -p /opt/unet-studio \
  && cd /opt/unet-studio \
  && mv /opt/src/build/unet_studio . \
  && chmod 755 unet_studio \
  && git clone https://github.com/frankyeh/UNet-Studio-Data.git \
  && rm -fr UNet-Studio-Data/.git \
  && mv UNet-Studio-Data/* . \
  && rm -fr UNet-Studio-Data

ENV PATH="$PATH:/opt/unet-studio" 

#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/unet-studio /
