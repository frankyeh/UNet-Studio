@echo off

set cuda_url=https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe
set cuda_installer=cuda_installer.exe

rem Download CUDA installer
curl -L -o %cuda_installer% %cuda_url%

rem Run CUDA installer in silent mode
start /wait %cuda_installer% --silent --toolkit --override --no-window --installpath=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

rem Clean up installer file
del %cuda_installer%

echo CUDA installation complete.