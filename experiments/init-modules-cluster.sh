# load modules
module load python/3 cuda/8.0 intel
# add CUDNN library
export LD_LIBRARY_PATH=/home/ih68sexe/cudnn/cuda/lib64:${LD_LIBRARY_PATH}
# suppress TF warning about missing CPU features (Your CPU supports instructions that this TensorFlow binary
# was not compiled to use: SSE4.1 SSE4.2 AVX)
# which is irrelevant here anyway as we use GPU
# export TF_CPP_MIN_LOG_LEVEL=2
