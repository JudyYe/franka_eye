# Install basics
- Specification
    + for GPUs like RTX 5080 (sm_120), you need cuda-toolkit > 12.6
    + python >= 3.10.0 (required by Grounded-SAM)


- pytorch
sm_120 requires torch cuda>=128
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Foundation Pose
Follow install instruction from their README
```
pip install -r requirements.txt

conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```


## GroundedSAM
```
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd grounding_dino
pip install -r requirement.txt
```


## FAQ
no module named torch during build: `setuptools` problem
```
"setuptools>=62.3.0,<75.9" [numpy==1.26.4]
```