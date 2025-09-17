CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=True python train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:31184 --resume \
--config-file ./configs/RSICD.yaml \
MODEL.F_NUM 3 \
DATALOADER.ENTITY_NUM 6 \
MODEL.CLIP_S True \
MODEL.CLIP_E True \
OUTPUT_DIR ./RS_work_dirs/RSICD_V"$j"_l2_2

