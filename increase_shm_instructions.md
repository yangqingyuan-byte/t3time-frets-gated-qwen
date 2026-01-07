# 增加 Docker 容器共享内存大小的方法

## 方法1：使用 docker run 命令（如果容器是通过 docker run 启动的）

在宿主机上执行以下命令来重新启动容器，并设置共享内存大小为 2GB：

```bash
# 1. 先停止当前容器（如果需要）
docker stop 66eccaa68c20

# 2. 重新启动容器，添加 --shm-size 参数
docker start 66eccaa68c20 --shm-size=2g

# 或者如果是 docker run 启动的，需要：
docker run --shm-size=2g [其他参数] [镜像名]
```

## 方法2：使用 docker-compose.yml（如果使用 docker-compose）

在 `docker-compose.yml` 文件中添加 `shm_size` 配置：

```yaml
version: '3'
services:
  your_service:
    image: your_image
    shm_size: '2gb'  # 设置共享内存为 2GB
    # 其他配置...
```

然后重新启动：
```bash
docker-compose down
docker-compose up -d
```

## 方法3：临时解决方案（在容器内，需要特权模式）

如果容器有特权模式，可以在容器内执行：
```bash
mount -o remount,size=2G /dev/shm
```

但当前容器没有足够权限，所以需要在宿主机上操作。

## 推荐的共享内存大小

- **最小**: 1GB (适合少量并行实验)
- **推荐**: 2GB (适合中等并行)
- **最佳**: 4GB+ (适合大量并行实验)

对于你的场景（8张GPU，每张GPU并行1-2个实验），建议设置为 **2GB**。

## 验证方法

在容器内执行：
```bash
df -h /dev/shm
```

应该显示类似：
```
Filesystem      Size  Used Avail Use% Mounted on
shm             2.0G  46M  1.9G   3% /dev/shm
```
