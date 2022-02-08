# 一、修改软件镜像

1、 
备份 /etc/apt/sources.list 文件
命令：`cp sources.list sources.list.back`

2、
编辑 sources.list 文件
命令：`vi sources.list`
阿里巴巴镜像源站：https://developer.aliyun.com/mirror/
将以下内容覆盖 sources.list 原内容：

```
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
 
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
 
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
 
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
 
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
```
3、
更新源
命令：`sudo apt-get update`

4、
更新软件
命令：`sudo apt-get upgrade`
