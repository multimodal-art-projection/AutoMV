export CUDA_VISIBLE_DEVICES=3
echo "use gpu ${CUDA_VISIBLE_DEVICES}"

export PYTHONPATH=../third_party:$PYTHONPATH

export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python infer/infer.py \
-i /map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/songformer/all_files.scp \
-o /map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/try \
--model SongFormer \
--checkpoint SongFormer.safetensors \
--config_path SongFormer.yaml \
-gn 1 \
-tn 1
# --debug
# --no_rule_post_processing