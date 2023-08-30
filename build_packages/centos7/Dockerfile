FROM unetstudio/centos7_qt5 as builder-stage

SHELL [ "/usr/bin/scl", "enable", "devtoolset-9"]
RUN gcc --version

RUN mkdir /opt/unet-studio \
  && cd /opt/unet-studio \
  && git clone https://github.com/frankyeh/UNet-Studio.git \
  && mv UNet-Studio src \
  && git clone https://github.com/frankyeh/TIPL.git \
  && mv TIPL src/TIPL \
  && mkdir -p /opt/unet-studio/build \
  && cd /opt/unet-studio/build \
  && qmake ../src/unet_studio.pro \
  && make

RUN cd /opt/unet-studio \
  && mv build/unet_studio . \
  && chmod 755 unet_studio \
  && rm -rf src build \

RUN curl -sL https://github.com/probonopd/linuxdeployqt/releases/download/7/linuxdeployqt-7-x86_64.AppImage > linuxdeployqt \
  && chmod a+x linuxdeployqt \
  && ./linuxdeployqt --appimage-extract \
  && ./squashfs-root/AppRun /opt/unet-studio/unet_studio -unsupported-bundle-everything -no-translations -bundle-non-qt-libs \
  && rm -fr squashfs-root \
  && rm -fr linuxdeployqt

ENV PATH="$PATH:/opt/unet-studio" 

#Create an empty container and transfer only the compiled software out
FROM scratch
COPY --from=builder-stage /opt/unet-studio /
