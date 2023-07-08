if [ $# -eq 3 ]; then
    # Number of computing devices.
    # devices=({"$1"//","/ })
    # num_trainers=${#devices[@]}
    echo "CUDA_VISIBLE_DEVICES: $1; experiments: $2; $3 trainers."
    CUDA_VISIBLE_DEVICES=$1 \
    python -m torch.distributed.run --nproc_per_node=$3\
             tools/main.py +experiments=$2
else
    echo "No arguments provided"
    exit 1
fi