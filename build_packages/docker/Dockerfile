FROM dsistudio/ubuntu2204_qt6_cuda:latest as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# libtorch
RUN apt update && apt full-upgrade -y && apt install -y --no-install-recommends python3-dev python3-pip python3-venv

# RUN pip3 install torch>=1.9.0+cpu torchvision>=0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

RUN cd /opt \
  && git clone https://github.com/frankyeh/UNet-Studio.git \
  && mv UNet-Studio src \
  && cd /opt/src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mkdir -p /opt/src/build

RUN cd /opt/src \
  && export TORCH_DIR="$(pip3 show torch | grep 'Location:' | awk '{print $2}')" \
  && cmake -S . -B build -GNinja -DCMAKE_PREFIX_PATH=$TORCH_DIR/torch -DCMAKE_BUILD_TYPE:STRING=Release -DTIPL_DIR=. \
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



