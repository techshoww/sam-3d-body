# 运行说明

## 使用 Docker Compose 启动

1. 设置模型路径环境变量（使用绝对路径）：

   ```bash
   export SAM3D_CHECKPOINT_DIR=/home/lihongjie/code/sam-3d-body-vith
   export SAM3D_CHECKPOINT_FILE=model.ckpt
   ```

2. 启动服务（首次会构建镜像）：

   ```bash
   docker compose up --build
   ```

3. 健康检查：

   ```bash
   curl http://127.0.0.1:8000/health
   ```

4. 运行客户端示例（在宿主机）：

   ```bash
   python client_example.py
   ```

## 使用 Docker 直接启动

```bash
docker build -t sam-3d-body .
docker run --gpus all -p 8000:8000 \
  -e SAM3D_CHECKPOINT_DIR=/abs/path/to/checkpoint_dir \
  -e SAM3D_CHECKPOINT_FILE=model.ckpt \
  -e SAM3D_MHR_DIR=/abs/path/to/mhr_dir \
  -e SAM3D_MHR_FILE=mhr_model.pt \
  -e SAM3D_CONFIG_PATH=/abs/path/to/model_config.yaml \
  -v /abs/path/to/checkpoint_dir:/abs/path/to/checkpoint_dir:ro \
  -v /abs/path/to/mhr_dir:/abs/path/to/mhr_dir:ro \
  -v /abs/path/to/model_config.yaml:/abs/path/to/model_config.yaml:ro \
  sam-3d-body
```

## 常见报错排查

- `Connection refused`：服务未启动或端口未映射，确保容器在跑且 `-p 8000:8000` 生效，访问 `http://127.0.0.1:8000/health` 验证。
- `SAM3D_CHECKPOINT_PATH is required`：未设置环境变量或路径错误，确认容器内可访问该路径。
- `model_config.yaml` 找不到：设置 `SAM3D_CONFIG_PATH` 指向配置文件并挂载到容器内。
- `checkpoint` 或 `mhr` 文件找不到：确认已挂载目录，并设置 `SAM3D_CHECKPOINT_DIR` / `SAM3D_MHR_DIR` 和对应文件名。
- `CUDA not available` 或 GPU 不可用：确认已安装 NVIDIA 驱动与容器工具链，启动命令带 `--gpus all` 或 compose 启用 GPU。
- `file not found`：检查模型文件在宿主机路径是否存在，并正确挂载到容器中。
