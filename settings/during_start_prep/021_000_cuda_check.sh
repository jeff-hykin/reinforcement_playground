# if nvcc exists
if [ -n "$(command -v "nvcc")" ]
then
    echo nvcc exists
    export __ENABLE_CUDA_FOR_FORNIX="true"
fi