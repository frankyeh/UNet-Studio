FROM dsistudio/ubuntu2404_qt6_cuda:latest as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# libtorch
RUN apt update && apt full-upgrade -y && apt install -y --no-install-recommends python3-dev python3-pip python3-venv

# RUN pip3 install torch>=1.9.0+cpu torchvision>=0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch torchvision torchaudio 

RUN cd /opt \
  && git clone https://github.com/frankyeh/UNet-Studio.git \
  && mv UNet-Studio src \
  && cd /opt/src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mkdir -p /opt/src/build

RUN cd /opt/src \
  && export TORCH_DIR="$(pip3 show torch | grep 'Location:' | awk '{print $2}')" \
  && cmake -S . -B build -GNinja -DCMAKE_PREFIX_PATH=${TORCH_DIR}/torch -DCMAKE_BUILD_TYPE:STRING=Release -DTIPL_DIR=. \
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
