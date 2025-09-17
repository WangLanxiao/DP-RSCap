# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import cv2
from torch import nn
import json
from PIL import Image
from tqdm import tqdm
import open_clip
import numpy as np
from torchvision import transforms
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from .mscoco import MSCoCoDataset
from ..build import DATASETS_REGISTRY
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data.transforms import _pil_interp
from timm.data.transforms import str_to_pil_interp as _pil_interp
import augly.image as imaugs

__all__ = ["MSCoCoEnd2EndDatasetRS"]

@DATASETS_REGISTRY.register()
class MSCoCoEnd2EndDatasetRS(MSCoCoDataset):
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        id2img_file: str,
        feats_folder: str,
        relation_file: str,
        imgorigin_path: str,
        gv_feat_file: str,
        attribute_file: str,
        sample_prob: float,
        augment_dp: float,
        input_size: int,
        entity_number: int,
    ):
        super(MSCoCoEnd2EndDatasetRS, self).__init__(
            stage,
            anno_file,
            seq_per_img, 
            max_feat_num,
            max_seq_len,
            feats_folder,
            relation_file,
            gv_feat_file,
            attribute_file,
            input_size
        )
        self.image_ids_path = id2img_file
        with open(self.image_ids_path, 'r') as f:
            self.ids2path = json.load(f)
        self.imgorigin_path=imgorigin_path
        self.sample_prob = sample_prob
        self.augment_dp = augment_dp
        self.entity_number = entity_number
        # 构建图像预处理单元
        model_name = 'ViT-B-32'  # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
        _, _, self.preprocess = open_clip.create_model_and_transforms(model_name)


        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=_pil_interp('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )
        # PreprocessCfg(size=(224, 224), mode='RGB', mean=(0.48145466, 0.4578275, 0.40821073),
        #               std=(0.26862954, 0.26130258, 0.27577711), interpolation='bicubic', resize_mode='shortest',
        #               fill_color=0)

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TRAIN),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER_VAL),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TEST)
        }
        id2img_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TRAIN_ID),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER_VAL_ID),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TEST_ID)
        }
        ret = {
            "stage": stage,
            "id2img_file": id2img_files[stage],
            "augment_dp":  cfg.DATALOADER.AUGMENT,
            "anno_file": ann_files[stage],
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "relation_file": cfg.DATALOADER.RELATION_FILE,
            "gv_feat_file": cfg.DATALOADER.GV_FEAT_FILE,
            "attribute_file": cfg.DATALOADER.ATTRIBUTE_FILE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "sample_prob": cfg.DATALOADER.SAMPLE_PROB,
            "input_size": cfg.DATALOADER.INPUT_SIZE,
            "imgorigin_path": cfg.DATALOADER.IMG_PATH,
            "entity_number": cfg.DATALOADER.ENTITY_NUM,
        }
        return ret

    def augly_augmentation(self, aug_image):
        aug = [
            # imaugs.blur(aug_image, radius=random.randint(1, 2)),
            imaugs.brightness(aug_image, factor=random.uniform(0.5, 1.5)),
            # imaugs.change_aspect_ratio(aug_image, ratio=random.uniform(0.8,1.5)),
            # imaugs.color_jitter(aug_image, brightness_factor=random.uniform(0.8,1.5), contrast_factor=random.uniform(0.8,1.5), saturation_factor=random.uniform(0.8,1.5)),
            # imaugs.crop(aug_image, x1=random.uniform(0,0.1), y1=random.uniform(0,0.1), x2=random.uniform(0.9,1), y2=random.uniform(0.9,1)),
            imaugs.hflip(aug_image),
            imaugs.vflip(aug_image),
            # imaugs.opacity(aug_image, level=random.uniform(0.5,1)),
            # imaugs.pixelization(aug_image, ratio=random.uniform(0.5,1)),
            # imaugs.random_noise(aug_image),
            imaugs.rotate(aug_image, degrees=random.randint(90, 90)),
            # imaugs.shuffle_pixels(aug_image, factor=random.uniform(0, 0.1)),
            # imaugs.saturation(aug_image, factor=random.uniform(1, 1.5)),
            imaugs.contrast(aug_image, factor=random.uniform(1, 1.5)),
            # imaugs.grayscale(aug_image)
        ]
        return random.choice(aug)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']    # dict {image_id: image_path}
        # image_id ='11209'    # dict {image_id: image_path}
        if len(self.feats_folder) > 0:
            # 读取图像features，并进行预处理
            feat_path = os.path.join(self.feats_folder, dataset_dict['filename'].split('.')[0] + '.npy')
            content = read_np(feat_path)['features']
            att_feats = content[-1, 0:self.max_feat_num].astype('float32')  # 49 * 2048
            global_feat = att_feats.mean(-2)  # 1 * 512
            ret = {
                kfg.IDS: image_id,
                kfg.ATT_FEATS: att_feats,
                kfg.PYR_FEATS: content,
                kfg.GLOBAL_FEATS: global_feat
            }
        else:
            # 读取图像，并进行预处理
            image_path = self.imgorigin_path+dataset_dict['filename']
            # image_path = '/data1/dataset/crowdcaption/crowdhuman2021/00000000.jpg'
            img = cv2.imread(image_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # img=Image.open(image_path)
            img_input = self.preprocess(img)
            # img.close()

            # img = cv2.imread(image_path)
            # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # if (random.random() > self.augment_dp) and (self.stage == 'train') :
            #     img = self.augly_augmentation(img)

            # import skimage
            # I = skimage.io.imread(image_path)
            # sa_ne='/home/amax/Desktop/123123/'
            # img.save(os.path.join(sa_ne, "origin.jpg"))
            # for i in range(14):
            #     image = self.augly_augmentation(img,i)
            #     image = image.convert("RGB")
            #     image.save(os.path.join(sa_ne, "{}strong".format(i) + ".jpg"))
            # img_input = self.transform(img)  # [3, 384, 384]，图像
            ret = {
                kfg.IDS: image_id,
                kfg.IMG_INPUT: img_input,
                # 'IMG_PATH': image_path,
            }

        if self.stage != 'train':
            g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
            class_ids = [dataset_dict['class_id']]
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type, 'CLASS_ID': class_ids})
            dict_as_tensor(ret)
            entity_name = dataset_dict['entity_name'][:self.entity_number]
            prompt_head = 'There are'
            prompt_tail = ' in image.'
            discrete_prompt = ''
            for entity in entity_name:  # gpt2 in transformer encoder ' ' + word into one token by default
                discrete_prompt += ' ' + entity + ','  # ' person, dog, park,'
            discrete_prompt = discrete_prompt[:-1]  # ' person, dog, park'
            discrete_prompt = prompt_head + discrete_prompt + prompt_tail
            ret.update({
                'ENTITY_NAME': discrete_prompt,
            })
            return ret
        
        sent_num = len(dataset_dict['tokens_ids'])
        if sent_num >= self.seq_per_img:
            selects = random.sample(range(sent_num), self.seq_per_img)
        else:
            selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        tokens_ids = [ dataset_dict['tokens_ids'][i,1:].astype(np.int64) for i in selects ]
        target_ids = [ dataset_dict['target_ids'][i,:-1].astype(np.int64) for i in selects ]
        class_ids = [dataset_dict['class_id'] for i in selects]
        g_tokens_type = [ np.ones((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ]

        entity_name = dataset_dict['entity_name'][:self.entity_number]
        # mask_number = int(0.4*self.entity_number)
        # entity_selects = random.sample(range(self.entity_number), mask_number)

        prompt_head = 'There are'
        prompt_tail = ' in image.'
        discrete_prompt = ''
        for idx_en, entity in enumerate(entity_name):  # gpt2 in transformer encoder ' ' + word into one token by default
            discrete_prompt += ' ' + entity + ','  # ' person, dog, park,'
            # if idx_en in entity_selects:
            #     discrete_prompt += ' ' + 'UNK' + ','
            # else:
            #     discrete_prompt += ' ' + entity + ','  # ' person, dog, park,'
        discrete_prompt = discrete_prompt[:-1]  # ' person, dog, park'
        discrete_prompt = prompt_head + discrete_prompt + prompt_tail
        entity_id = dataset_dict['entity_id'][:self.entity_number]
        entity_score = dataset_dict['entity_score'][:self.entity_number]

        ret.update({
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
            'CLASS_ID': class_ids,
            'ENTITY_ID': entity_id,
            'ENTITY_SCORE': entity_score,
        })
        dict_as_tensor(ret)
        ret.update({
            'ENTITY_NAME': discrete_prompt,
        })
        return ret