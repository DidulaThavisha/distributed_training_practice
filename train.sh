set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/flux/run_train.sh
NGPU=${NGPU:-"2"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./config.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi


PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m train --job.config_file ${CONFIG_FILE} $overrides
