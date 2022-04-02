> https://blog.csdn.net/m0_37412775/article/details/109355044

# 设置python镜像源
不知道是不是因为学校就是USTC的原因，一开始把镜像设置成其他地方网速还是很慢。特别是安装pytorch的时候，因为官网的命令是

```pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113```

这好像会指定到安装路径还是国外的，然后速度又是很卡，这里使用命令来修改镜像

```
# 使用本镜像站来升级 pip
pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple pip -U
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```

# 安装Cuda

由于Ubuntu20.04自带gcc版本为9.7.0，而Cuda10.2需要添加并切换为gcc7。

```
sudo apt install gcc-7 g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 50
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
```

Ubuntu20.04应该是有nvidia驱动的，所以这里驱动先不用动，直接开始安装cuda
首先使用`nvidia-smi`命令得到的`CUDA Version: 11.4 `是支持的最大cuda版本，所以安装的cuda版本不应该大于这个。
在`https://developer.nvidia.com/cuda-toolkit-archive`可以找到对应版本的cuda
再依次点击 Linux  x86_64  Ubuntu  20.04  runfile(local) 
之后出现两个命令，一个是下载，另一个是安装
进入安装之后，第一个选择是continue，第二个选择需要把cuda自带的显卡驱动给去掉，也就是当选项在DRIVER的时候，按`空格`取消掉
然后才install安装

之后要修改变量
```
sudo vim ~/.bashrc \\进入vim界面。输入字母i，进入编辑模式
\\在bashrc文件中输入以下命令，注意修改你的cuda版本
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64
export PATH=$PATH:/usr/local/cuda-11.4/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.4
\\输入完成后，点击esc键并输入:wq!，再按esc键退出vim。
source ~/.bashrc \\运行.bashrc文件
```

`nvcc --version`显示正常就对了


# cudann安装
版本需要对应，和cuda版本要对应

`https://developer.nvidia.com/rdp/cudnn-archive`

选择对应版本的`Linux(x86_64)'版本
下载完后解压出cuda文件夹
```
$sudo cp cuda/include/cudnn*.h /usr/local/cuda/include/
 
$sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
 
$sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```
三个命令输完可能就没问题



# 问题
我在运行pytorch测试程序的报错
```
UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:112.)
```
查到的用这个命令就解决了问题
`apt-get install nvidia-modprobe`


