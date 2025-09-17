j=VV6_2

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:31184 \
#--config-file ./configs/image_caption/baseline/Sydney_rl.yaml \
#MODEL.WEIGHTS ./RS_work_dirs/Sydney_VRS1_modified_vgg16/model_Epoch_00023_Iter_0000160.pth \
#OUTPUT_DIR ./RS_work_dirs/Sydney_V"$j"_RL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=True python train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:13184  \
--config-file ./configs/UCM_rl.yaml \
MODEL.WEIGHTS ./RS_work_dirs/UCM_Vmodified_en6_imdi1024/model_Epoch_00012_Iter_0000155.pth \
OUTPUT_DIR ./RS_work_dirs/UCM_Vmodified_en6_imdi1024_RL

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:21184 \
#--config-file ./configs/image_caption/baseline/RSICD_rl.yaml \
#MODEL.WEIGHTS ./RS_work_dirs/RSICD_VRS1_modified_vgg16/model_Epoch_00006_Iter_0000407.pth \
#OUTPUT_DIR ./RS_work_dirs/RSICD_V"$j"_RL