curl -o pytorch.zip "https://download.pytorch.org/libtorch/cu117/libtorch-win-shared-with-deps-2.0.0%%2Bcu117.zip"
tar -xf pytorch.zip -C . --strip-components 1 libtorch/lib
del pytorch.zip
xcopy /Y /S lib\*.dll .
rmdir /s /q lib

curl -L -o cuda_installer.exe https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_516.01_windows.exe
start /wait cuda_installer.exe -s
del cuda_installer.exe