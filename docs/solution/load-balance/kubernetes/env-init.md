# 环境准备

本节介绍集群节点的基础环境准备工作操作。

> [!TIP]
> 🎯 所有操作面向集群所有节点

## 设置主机名

为集群节点设置主机名，确保可以通过主机名进行通信。

可通过 `hostnamectl` 命令查看当前主机名，如果符合预期，可跳过当前步骤。

```bash
# 编辑 hostname文件
vim /etc/hostname
# 填写主机名
k8s-master
```

主机名修改后请重启主机，确保主机名生效。

## hosts 解析

将需要通信的所有节点配置在各自的 hosts 文件中。

```bash
echo -e "192.168.1.4 k8s-master\n192.168.1.5 k8s-worker\n" | sudo tee -a /etc/hosts
```

## 关闭防火墙服务

1. 关闭防火墙服务

    ```bash
    sudo ufw disable
    ```

    检查防火墙状态

    ```bash
    sudo ufw disable
    ```

    输出结果类似以下内容，表明已关闭：

    ```bash
    Status: inactive
    ```

2. 禁用防火墙服务，防止系统重启后自动启动

```bash
sudo systemctl stop ufw
sudo systemctl disable ufw
```

3. 验证防火墙状态

```bash
sudo systemctl status ufw
```

> [!TIP]
> 如后期需要开启防火墙，请在安装成功后开启，并公开相应的端口

## 永久关闭 selinux

selinux 负责 Linux 在系统底层资源调用的安全管控，在使用 Kubernetes 时需要关闭。

Ubuntu 默认不启用 selinux，可跳过此步骤。

1. 编辑 SELinux 配置文件

```bash
sudo vi /etc/selinux/config

# 修改 SELINUX=enforcing 为 SELINUX=disabled
SELINUX=disabled
```

2. 保存并重启系统。系统重启成功后验证 SELinux 状态：

```bash
sestatus
```

## 关闭 swap

swap 会影响系统性能，因此 Kubernetes 官方推荐关闭 swap，否则初始化控制面板时会强制停止。

只需打开 `/etc/fstab`，将如下行禁用即可

```bash
#/swap.img none swap sw 0 0
```

可使用命令 `free -h` 查看。如果 swap 都是 0 表明系统停止 swap。

## 系统时间同步

集群服务需要多个节点的时间保持一致。

可通过 `date` 或者 `timedatectl` 命令查看当前节点的时间。

如果时间不正确，请替换为东八区-上海

```bash
timedatectl set-timezone Asia/Shanghai
```

## 确认网络通信的模块开启

```bash
modprobe overlay
modprobe br_netfilter
```

## 配置流量链

有一些 ipv4 的流量不能走 iptables 链，因为 linux 内核的一个过滤器，每个流量都会经过他，然后再匹配是否可进入当前应用进程去处理，所以会导致流量丢失。

1. 配置 k8s.conf 文件（k8s.conf 文件本身不存在，需要自己创建的）

```bash
sudo vim /etc/sysctl.d/k8s.conf

# 新增如下内容
vm.swappiness=0
net.bridge.bridge-nf-call-ip6tables=1
net.bridge.bridge-nf-call-iptables=1
net.ipv4.ip_forward=1
```

2. 执行如下命令重载系统参数

```bash
sudo sysctl --system
```

3. 验证结果

```bash
# 此命令结果必须为 1
cat /proc/sys/net/ipv4/ip_forward
```
