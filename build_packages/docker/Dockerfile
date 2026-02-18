FROM dsistudio/ubuntu2204_qt6_cuda:latest

ENV DEBIAN_FRONTEND=noninteractive
# Add the final binary to PATH so you can run 'unet_studio' from anywhere
ENV PATH="/opt/unet-studio:$PATH"

# 1. Install System Dependencies & PyTorch
# We combine updates and installs to keep the layer clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    libcudnn8-dev \
    libsndfile1 \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir \
       torch==1.13.0+cu117 \
       torchvision==0.14.0+cu117 \
       torchaudio==0.13.0 \
       --extra-index-url https://download.pytorch.org/whl/cu117

# 2. Build Source
WORKDIR /opt/src
RUN git clone https://github.com/frankyeh/UNet-Studio.git . \
    && git clone https://github.com/frankyeh/TIPL.git \
    # Get the torch path cleanly using python
    && TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))") \
    && cmake -S . -B build -GNinja \
       -DCMAKE_PREFIX_PATH="${TORCH_PATH}" \
       -DCMAKE_BUILD_TYPE=Release \
       -DTIPL_DIR=$(pwd)/TIPL \
    && cmake --build build --parallel

# 3. Setup Final Directory & Data
WORKDIR /opt/unet-studio
RUN cp /opt/src/build/unet_studio . \
    && git clone https://github.com/frankyeh/UNet-Studio-Data.git temp_data \
    && mv temp_data/* . \
    && rm -rf temp_data /opt/src
