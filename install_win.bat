set url=https://download.pytorch.org/libtorch/cu117/libtorch-win-shared-with-deps-2.0.0%%2Bcu117.zip
curl -o pytorch.zip "%url%"
del pytorch.zip
xcopy /Y /S lib\*.dll .
rmdir /s /q lib
