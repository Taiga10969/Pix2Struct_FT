from models.pix2struct.processing_pix2struct import Pix2StructProcessor
from models.pix2struct.modeling_pix2struct import Pix2StructForConditionalGeneration
from models.t5.tokenization_t5_fast import T5TokenizerFast 
from models.pix2struct.image_processing_pix2struct import Pix2StructImageProcessor


import cv2
import time
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')

model_name = 'google/deplot'

image_processor = Pix2StructImageProcessor.from_pretrained(model_name)
tokenizer = T5TokenizerFast.from_pretrained(model_name)
processor = Pix2StructProcessor(image_processor = image_processor,
                                tokenizer = tokenizer)
processor.image_processor.is_vqa = False # タスクをvqa=TueからFalseに変更
model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to(device)


trained_model_pth = './DePlot_FT_param/run_result_FT_test_001_epoch_10.pth'
msg = model.load_state_dict(torch.load(trained_model_pth, map_location=torch.device(device)))
print(f"model.load_state_dict [info] msg : {msg}")

img_pth =  './demo_image/flamingo.png'
image = Image.open(img_pth).convert('RGB')

input_prompt = 'Caption for this figure TEST : '

inputs, info = processor(images=image, text=input_prompt, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

predictions = model.generate(**inputs, 
                             output_attentions = True,
                             output_hidden_states = True,
                             max_new_tokens = 129,
                             return_dict_in_generate = True,
                             )

#print(predictions.keys()) # odict_keys(['sequences', 'encoder_attentions', 'encoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'decoder_hidden_states'])


print("predictions.sequences : ", processor.decode(predictions['sequences'][0], skip_special_tokens=True))
print("predictions.tokens", tokenizer.convert_ids_to_tokens(predictions['sequences'][0]))

print("len(predictions['encoder_attentions']) : ", len(predictions['encoder_attentions'])) # 12 << (layer)
print("predictions['encoder_attentions'][0].shape : ", predictions['encoder_attentions'][0].shape) # torch.Size([1, 12, 2048, 2048])


# 可視化コード
batch = 0
plt_save_pormat = 'png'

encoder_attentions = []
for layer in range(0, len(predictions['encoder_attentions']), 1):
    encoder_attentions.append(predictions['encoder_attentions'][layer][batch].to(torch.float).cpu().detach().numpy())
#print("np.shape(encoder_attentions) : ", np.shape(encoder_attentions)) # (12, 12, 2048, 2048) << (layer, head, h, w)


# Pix2StructVisionModelのAttentio Weightの可視化（最終層のヘッド平均したものを可視化）
layer = -1
plt.imshow(np.mean(encoder_attentions[layer], axis=0))
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'./get_encoder_attention/pix2struct_VisionModel_{layer}layer_mean_head.{plt_save_pormat}', transparent=True)
plt.close()

# Attention rolloutでの可視化
encoder_attention_rollout = utils.Attention_Rollout(encoder_attentions)
plt.imshow(encoder_attention_rollout)
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'./get_encoder_attention/pix2struct_VisionModel_attention_rollout.{plt_save_pormat}', transparent=True)
plt.close()


# processorが出力する画像情報の詳細(info)を取得
patch_nums = info['extract_flattened_patches_info'][batch]
patch_num_rows = patch_nums['rows']         # パッチの行と列の数
patch_num_columns = patch_nums['columns']   # パッチの行と列の数 
active_patch_num = patch_num_rows * patch_num_columns
print(f'processor [info] \npatch_num_rows = {patch_num_rows}\npatch_num_columns = {patch_num_columns}\nactive_patch_num = {active_patch_num}')

view_patch_token = 0

attention = encoder_attention_rollout[:active_patch_num, :active_patch_num] # processorでのpadding部分を削除
attention = attention[view_patch_token]
attention = np.reshape(attention, (patch_num_rows, patch_num_columns)) # 画像のサイズにreshape 縦横で取るpatch数が異なる．


# 画像サイズに拡大し，RGB3次元画像に変換し，画像と重ね合わせる
# 入力画像の可視化
#print('type(img) : ', type(img)) # <class 'PIL.Image.Image'>
img_width, img_height = image.size

attention_map = cv2.resize(attention, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
attention_map_rgb = utils.heatmap_to_rgb(attention_map)
attention_map = cv2.addWeighted(np.array(image), 0.25, attention_map_rgb, 0.75, 0)

plt.imshow(attention_map)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'./get_encoder_attention/attention_map.{plt_save_pormat}')
plt.close()
