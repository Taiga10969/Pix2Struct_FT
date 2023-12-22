import os
import re
import time
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SciCapPlusDataset(Dataset):
    def __init__(self,
                 scicap_plus_dataset_path,
                 is_train,
                 contains_subfigure = False,
                 include_val = False,
                 generate = False,
                 template = "Caption for this figure : ",
                 check = False,
                 ):
        super().__init__()
        start_time = time.time()

        self.dataset_path = scicap_plus_dataset_path
        self.train = is_train
        self.contains_subfigure = contains_subfigure
        self.include_val = include_val
        self.generate = generate
        self.prompt_template = template
        self.check = check

        self.image_filenames = self._get_image_list()
        self.id_abstract_data = self.load_json_file(os.path.join(self.dataset_path, "scicap_data/id_abstract_dict.json"))

        #self.pattern = re.compile(r'imgs/(train/\S+)\.png')
        self.pattern = re.compile(r'imgs/(\w+)/(\S+)\.png')
        
        if self.check == True:
            self.not_find_abstract_list = []
            self.cut_mention_text_idx_list = []
            self.cut_caption_text_idx_list = []
            self.mention_text_len_list = []
            self.caption_text_len_list = []
            self.non_mention_text_list = []

        end_time = time.time()

        print(f'SciCapPlusDataset [info] : end of __init__ {end_time - start_time}s  len(dataset)={len(self.image_filenames)}')
    
    def _get_image_list(self):
        
        if self.train == True:
            file_path = os.path.join(self.dataset_path, "image_list", "image_file_path_list_train.json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            
            image_list = json_data.get("contains_subfigure_false", [])

            if self.contains_subfigure == True:
                image_list += json_data.get("contains_subfigure_true", [])
        
        else:
            file_path = os.path.join(self.dataset_path, "image_list", "image_file_path_list_test.json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            
            image_list = json_data.get("contains_subfigure_false", [])

            if self.contains_subfigure == True:
                image_list += json_data.get("contains_subfigure_true", [])

        if self.include_val == True:
            file_path = os.path.join(self.dataset_path, "image_list", "image_file_path_list_val.json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            
            image_list += json_data.get("contains_subfigure_false", [])

            if self.contains_subfigure == True:
                image_list += json_data.get("contains_subfigure_true", [])

        return image_list

    def load_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
       
    def paperID_delete_version_information(self, paperId:str):
        # 入力されたpaperIdにversion情報(vの文字)が含まれているかを確認し，含まれている場合はvを含むそれ以降の文字列を削除する
        if 'v' in paperId:
            paperId = paperId.split('v')[0]
        else:
            print(f"paper-ID : {paperId}はversion情報を含みません.")
        
        return paperId
    
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]# + '.png'
        img_path = os.path.join(self.dataset_path, img_path)

        #if self.image_processor:
        #    img = Image.open(img_path).convert('RGB')
        #    img, info = self.image_processor(img)
        
        # 画像パスからcaptionデータを取得する
        match = self.pattern.search(img_path)

        caption_data_path = os.path.join(self.dataset_path, "captions", match.group(1) + '/' + match.group(2) + ".json")
        data = self.load_json_file(caption_data_path)
        #caption = data.get("1-lowercase-and-token-and-remove-figure-index")["caption"]  # キーが"1-lowercase-and-token-and-remove-figure-index"の値を取得
        caption = data.get("2-normalized")["2-2-advanced-euqation-bracket"]["caption"]  # キーが"2-normalized" >> "2-2-advanced-euqation-bracket"の値を取得
        
        if self.check == True:
            #self.mention_text_len_list.append(len(mention_text))
            self.caption_text_len_list.append(len(caption))


        if self.generate == False:
            prompt = self.prompt_template + caption
            return img_path, prompt
        
        if self.generate == True:
            prompt = self.prompt_template
            return img_path, caption, prompt


if __name__ == '__main__':

    # dataset test
    dataset = SciCapPlusDataset(scicap_plus_dataset_path="/taiga/Datasets/scicap_plus",
                                is_train = True,
                                contains_subfigure = False,
                                include_val = False,
                                generate= False,
                                template="Caption for this figure : ",
                                )
    
    img_path, prompt = dataset[0]
    print('img_path : ', img_path)
    print('prompt : ', prompt)


    # dataloarder test
    import torch
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=4,
                                              shuffle=True,
                                              )
    
    # processor test
    from models.t5.tokenization_t5_fast import T5TokenizerFast 
    from models.pix2struct.image_processing_pix2struct import Pix2StructImageProcessor
    from models.pix2struct.processing_pix2struct import Pix2StructProcessor
    
    device = 'cpu'

    model_name = 'google/deplot'

    image_processor = Pix2StructImageProcessor.from_pretrained(model_name)
    image_processor.is_vqa = False # vqaタスクの設定をvqa=TueからFalseに変更 >> これにより，"decoder_attention_mask", "decoder_input_ids"が追加される．

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    processor = Pix2StructProcessor(image_processor = image_processor,
                                    tokenizer = tokenizer)

    
    images, labels  = next(iter(data_loader))

    image_list = []
    for img_path in images:
        # 画像を開きRGB形式に変換
        img = Image.open(img_path).convert('RGB')
        image_list.append(img)
    
    inputs, info = processor(images=image_list,
                             #generation_config = None, 
                             text=labels, 
                             return_tensors="pt", 
                             max_length=128, 
                             truncation=True, 
                             padding="longest",
                             )
    
    #print("inputs : ", inputs)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    #print("inputs['flattened_patches'] : ", inputs['flattened_patches'])
    print("inputs['decoder_input_ids'] : ", inputs['decoder_input_ids'])