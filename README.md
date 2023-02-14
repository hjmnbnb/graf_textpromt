# GRAF_TEXTPROMT
#创建环境（pytorch >= 1.7.1，CUDA >= v10.1）

You can create an anaconda environment called `graf` using
```
conda env create -f environment.yml
conda activate graf
```
#安装nerf依赖和torchsearchsorted

Next, for nerf-pytorch install torchsearchsorted. Note that this requires `torch>=1.7.1` and `CUDA >= v10.1`.
You can install torchsearchsorted via
``` 
cd submodules/nerf_pytorch
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ../../../
```

#安装CLIP
```
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

#训练demo模型
```
python test1.py configs/carla.yaml
```
