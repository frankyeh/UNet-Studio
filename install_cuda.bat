curl -L -o cuda_installer.exe https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_516.01_windows.exe
start /wait cuda_installer.exe --silent --toolkit --override --no-window --installpath=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
del cuda_installer.exe

