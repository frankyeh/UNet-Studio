FROM dsistudio/ubuntu2204_qt6:latest as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# libtorch
RUN apt install -y --no-install-recommends python3-dev python3-pip python3-venv
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN python3 -m torch.utils.collect_env

RUN cd /opt \
  && git clone https://github.com/frankyeh/UNet-Studio.git \
  && mv UNet-Studio src \
  && cd /opt/src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mkdir -p /opt/src/build

RUN cd /opt/src \
  && cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE:STRING=Release -DTIPL_DIR=. \
  && cmake --build ./build --parallel --config Release

RUN mkdir -p /opt/unet-studio \
  && cd /opt/unet-studio \
  && mv /opt/src/build/unet_studio . \
  && chmod 755 unet_studio \
  && git clone https://github.com/frankyeh/UNet-Studio-Network.git \
  && rm -fr UNet-Studio-Network/.git \
  && mv UNet-Studio-Network/* . \
  && rm -fr UNet-Studio-Network

ENV PATH="$PATH:/opt/unet-studio" 



