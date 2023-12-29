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
from tqdm import tqdm

import utils
from configs import generate_text_Config

config = generate_text_Config()

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')



image_processor = Pix2StructImageProcessor.from_pretrained(config.pretrained_model_name)
tokenizer = T5TokenizerFast.from_pretrained(config.pretrained_model_name)
processor = Pix2StructProcessor(image_processor = image_processor,
                                tokenizer = tokenizer)

model = Pix2StructForConditionalGeneration.from_pretrained(config.pretrained_model_name).to(device)
model = model.eval()

print("processor.image_processor.is_vqa : ", processor.image_processor.is_vqa)
processor.image_processor.is_vqa = False # タスクをvqa=TueからFalseに変更
print("processor.image_processor.is_vqa : ", processor.image_processor.is_vqa)

trained_model_pth = '/taiga/Pix2Struct_FT_captioning/run_result_matcha_base_train_figure_captioning_001/run_result_matcha_base_train_figure_captioning_001_epoch_10.pth'
msg = model.load_state_dict(torch.load(trained_model_pth, map_location=torch.device(device)))
print(f"model.load_state_dict [info] msg : {msg}")


# OUTPUT_ATTENTION_CONFIG ###############
#img_pth =  './demo_image/flamingo.png'
img_pth =  './demo_image/LLaMA.png'
input_prompt = 'Caption for this figure : '
##########################################

image = Image.open(img_pth).convert('RGB')

width, height = image.size
print(f'image.size : width {width}, height {height}')

inputs, info = processor(images=image, text=input_prompt, return_tensors="pt")

inputs = {key: value.to(device) for key, value in inputs.items()}
# decoder_input_idsから<eos>を削除
if processor.image_processor.is_vqa == False:
    inputs['decoder_input_ids'] = inputs['decoder_input_ids'][:, :-1]
    inputs['decoder_attention_mask'] = inputs['decoder_attention_mask'][:, :-1]
print(inputs)
    
# 入力画像を調査 0番目のデータに対して
input_image = info['input_image'][0]
input_image = Image.fromarray(input_image).convert('RGB')
width, height = input_image.size
print(f'input_image.size : width {width}, height {height}')
input_image.save('./get_cross_attention/input_image.png')
    
#print(inputs)

predictions = model.generate(flattened_patches = inputs['flattened_patches'],
                             attention_mask = inputs['attention_mask'],
                             output_attentions = True,
                             output_hidden_states = True,
                             max_new_tokens = 128,
                             return_dict_in_generate = True,
                             )

predict_text = processor.decode(predictions['sequences'][0], skip_special_tokens=True)
print("predict_text : ", predict_text)

predict_token = tokenizer.convert_ids_to_tokens(predictions['sequences'][0])[1:]
print("len(predict_token) : ", len(predict_token))
print("predict_token", predict_token)
eos_token_index = predict_token.index('</s>')
print("eos_token_index : ", eos_token_index)

print("len(predictions['cross_attentions']) : ", len(predictions['cross_attentions'])) # 128 << (words)
print("len(predictions['cross_attentions'][0]) : ", len(predictions['cross_attentions'][-1])) # 12 << (layer)
print("predictions['cross_attentions'][0].shape : ", predictions['cross_attentions'][-1][0].shape) # torch.Size([1, 12, 1, 2048]) << (batch, head, h, w)


# OUTPUT_ATTENTION_CONFIG ###############
batch = 0
plt_save_pormat = 'svg'
num_columns = 5  # 単語毎のAttentionを一枚の画像にする際の横方向の画像枚数
##########################################

# processorが出力する画像情報の詳細(info)を取得
patch_nums = info['extract_flattened_patches_info'][batch]
patch_num_rows = patch_nums['rows']         # パッチの行と列の数
patch_num_columns = patch_nums['columns']   # パッチの行と列の数 
active_patch_num = patch_num_rows * patch_num_columns
print(f'processor [info] \npatch_num_rows = {patch_num_rows}\npatch_num_columns = {patch_num_columns}\nactive_patch_num = {active_patch_num}')
input_image = info['input_image'][0]
input_image = Image.fromarray(input_image).convert('RGB')
img_width, img_height = input_image.size

# cross_attentionの情報を収集
cross_attentions = []
for layer in range(0, len(predictions['cross_attentions'][-1]), 1):
    cross_attentions.append(predictions['cross_attentions'][-1][layer][batch].to(torch.float).cpu().detach().numpy())
print("np.shape(cross_attentions) : ", np.shape(cross_attentions)) # (layer, head, word, patch_token)


# head平均
head_mean_cross_attention = np.mean(cross_attentions, axis=1)
layer_mean_cross_attention = np.mean(head_mean_cross_attention, axis=0)

print("np.shape(layer_mean_cross_attention) : ", np.shape(layer_mean_cross_attention)) # (word, patch_token)


# ある単語のみを出力するためのプログラム
'''
view_word_idx = 7
attention = layer_mean_cross_attention[:active_patch_num, :active_patch_num] # processorでのpadding部分を削除
attention = attention[view_word_idx]
attention = np.reshape(attention, (patch_num_rows, patch_num_columns)) # 画像のサイズにreshape 縦横で取るpatch数が異なる．

attention_map = cv2.resize(attention, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
attention_map_rgb = utils.heatmap_to_rgb(attention_map)
attention_map = cv2.addWeighted(np.array(input_image), 0.25, attention_map_rgb, 0.75, 0)

plt.imshow(attention_map)
plt.title(predict_token[view_word_idx])
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'./get_cross_attention/attention_map_{view_word_idx}.{plt_save_pormat}', transrate=False)
plt.close()
'''

prediction_word_choose_list = utils.get_word_token_idx(predict_token, predict_text.split())
print("prediction_word_choose_list : ", prediction_word_choose_list)
print("prediction_word_choose_list.keys() : ", prediction_word_choose_list.keys())

active_cross_attention = layer_mean_cross_attention[:active_patch_num, :active_patch_num]

for view_word in tqdm(prediction_word_choose_list.keys(), desc="Processing_attention_map"):
    group_attntion = []
    for i in prediction_word_choose_list[view_word]:
        group_attntion.append(active_cross_attention[i])
    
    group_attntion = np.mean(group_attntion, axis=0)
    group_attntion = np.reshape(group_attntion, (patch_num_rows, patch_num_columns))

    group_attntion_map = cv2.resize(group_attntion, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    group_attntion_map_rgb = utils.heatmap_to_rgb(group_attntion_map)
    group_attntion_map = cv2.addWeighted(np.array(input_image), 0.25, group_attntion_map_rgb, 0.75, 0)

    plt.imshow(group_attntion_map)
    plt.title(view_word)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.savefig(f'./get_cross_attention/attention_map_{view_word}.{plt_save_pormat}',transparent=True)
    plt.close()




# 画像をまとめるキャンバスの初期化
fig, axs = plt.subplots(nrows=-(-len(prediction_word_choose_list)//num_columns), ncols=num_columns, figsize=(15, 8))

for idx, view_word in tqdm(enumerate(prediction_word_choose_list.keys()), desc="Processing_attention_maps"):
    group_attntion = []
    for i in prediction_word_choose_list[view_word]:
        group_attntion.append(active_cross_attention[i])

    group_attntion = np.mean(group_attntion, axis=0)
    group_attntion = np.reshape(group_attntion, (patch_num_rows, patch_num_columns))

    group_attntion_map = cv2.resize(group_attntion, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    group_attntion_map_rgb = utils.heatmap_to_rgb(group_attntion_map)
    group_attntion_map = cv2.addWeighted(np.array(input_image), 0.25, group_attntion_map_rgb, 0.75, 0)

    # 画像をキャンバスに配置
    axs.flat[idx].imshow(group_attntion_map)
    axs.flat[idx].set_title(utils.generate_title(view_word))
    axs.flat[idx].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
# 余分なサブプロットがあれば非表示にする
for idx in range(len(prediction_word_choose_list), axs.size):
    axs.flat[idx].axis('off')
# 余白を調整
plt.tight_layout()

# 画像を保存
plt.savefig(f'./get_cross_attention/attention_maps.{plt_save_pormat}', transparent=True)
plt.show()
