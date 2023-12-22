import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt

def torch_fix_seed(seed=1):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    print("utils [info] : torch_fix_seed")


def train_one_epoch(model, train_loader, processor, optimizer, config, device, wandb):
    
    # ネットワークモデルを学習モードに設定
    model.train()

    sum_loss = 0.0
    count = 0
    
    with tqdm(train_loader) as pbar:
        for images, labels in pbar:
            image_list = []
            for img_path in images:
                # 画像を開きRGB形式に変換
                img = Image.open(img_path).convert('RGB')
                image_list.append(img)
            
            inputs, _ = processor(images=image_list,
                                  text=labels, 
                                  return_tensors="pt", 
                                  max_length=config.max_length, 
                                  truncation=True, 
                                  padding="longest",
                                  )

            inputs = {key: value.to(device) for key, value in inputs.items()}
            count += 1

            outputs= model(flattened_patches = inputs['flattened_patches'],
                           attention_mask = inputs['attention_mask'],
                           #decoder_input_ids = inputs['decoder_input_ids'],
                           #decoder_attention_mask = inputs['decoder_attention_mask'],
                           labels = inputs['decoder_input_ids'],
                           )

            loss = outputs.loss.mean()
    
            optimizer.zero_grad()
            loss.backward()

            if config.clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)

            optimizer.step()
    
            sum_loss += loss.item()
    
            pbar.set_postfix(OrderedDict(loss=loss.item(), 
                                         ave_loss=sum_loss/count, 
                                         lr = optimizer.param_groups[0]['lr'],
                                         ))
            
            loss_num = loss.item()

            if config.wandb:
                wandb.log({
                    "train_iter_loss" : loss_num,
                    "lr" : optimizer.param_groups[0]['lr'],
                    })
    
    return sum_loss/count



def test_one_epoch(model, test_loader, processor, config, device, wandb):
    
    # ネットワークモデルを学習モードに設定
    model.eval()

    sum_loss = 0.0
    count = 0
    
    with tqdm(test_loader) as pbar:
        for images, labels in pbar:
            image_list = []
            for img_path in images:
                # 画像を開きRGB形式に変換
                img = Image.open(img_path).convert('RGB')
                image_list.append(img)
            
            inputs, _ = processor(images=image_list, 
                                  text=labels, 
                                  return_tensors="pt", 
                                  max_length=config.max_length, 
                                  truncation=True, 
                                  padding="longest",
                                  )

            inputs = {key: value.to(device) for key, value in inputs.items()}
            count += 1

            outputs= model(flattened_patches = inputs['flattened_patches'],
                           attention_mask = inputs['attention_mask'],
                           #decoder_input_ids = inputs['decoder_input_ids'],
                           #decoder_attention_mask = inputs['decoder_attention_mask'],
                           labels = inputs['decoder_input_ids'],
                           )

            loss = outputs.loss.mean()
       
            sum_loss += loss.item()
    
            pbar.set_postfix(OrderedDict(loss=loss.item(), 
                                         ave_loss=sum_loss/count, 
                                         ))
            
            loss_num = loss.item()

            if config.wandb:
                wandb.log({
                    "test_iter_loss" : loss_num,
                    })
    
    return sum_loss/count


def freeze_pix2struct_parameters(pix2struct_model):
    # Pix2StructVisionModelのパラメータを取得
    pix2struct_params = pix2struct_model.encoder.parameters()

    # パラメータを凍結
    for param in pix2struct_params:
        param.requires_grad = False
    print("freeze_pix2struct_parameters [info]: Completed freeze pix2struct_params")


def Attention_Rollout(vision_attns):
    mean_head = np.mean(vision_attns, axis=1)
    mean_head = mean_head + np.eye(mean_head.shape[1])
    mean_head = mean_head / mean_head.sum(axis=(1,2))[:, np.newaxis, np.newaxis]

    v = mean_head[-1]
    for n in range(1,len(mean_head)):
        v = np.matmul(v, mean_head[-1-n])
    
    return v


def heatmap_to_rgb(heatmap, cmap='jet'):
    # 0から1の範囲に正規化
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # カラーマップに変換
    colormap = plt.get_cmap(cmap)
    colored_heatmap = (colormap(normalized_heatmap) * 255).astype(np.uint8)
    # RGBに変換
    rgb_image = colored_heatmap[:, :, :3]
    
    return rgb_image