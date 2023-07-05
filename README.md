# tutorial_tensorflow
bunch of tensor flow tutorials

```
pip freeze > requirements.txt
conda create -n tflow
conda activate tflow
pip install -r requirements.txt
```
To install [TensorFlow](https://www.tensorflow.org/install/pip) with pip, if the requirements.txt doesn't work.

```
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

```
conda clean -tp  # delete tarballs and unused packages
conda remove --name myenv --all # To remove an environment
```