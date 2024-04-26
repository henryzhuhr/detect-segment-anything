# Detect-Segment-anything


## 创建环境

确保电脑安装了 conda ，如果没有请安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

获取相关资源执行：
```shell
bash scripts/get-resource.sh
# zsh for MacOS
```

安装虚拟环境至当前目录 `.env` 下，执行以下命令：
```shell
bash scripts/init-env.sh
```
- 脚本中 `CUSTOM_PYTHON_VERSION` 可以指定 python 版本，否则根据系统自带的 python 版本安装。
- 脚本会根据系统的 CUDA 版本安装对应的 pytorch 和 torchvision。需要设置 `CUDA_VERSION` ，否则只安装 CPU 版本。




## 快速上手


## 转换

### PyTorch 转化为 ONNX

```shell
python3 export.py
```



## 参考资料

- 🚀 [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- 🚀 [GroundingDINO(wenyi5608-openvino)](https://github.com/wenyi5608/GroundingDINO/tree/wenyi5608-openvino)
- TensorRT with GroundingDINO [#46](https://github.com/IDEA-Research/GroundingDINO/issues/46)