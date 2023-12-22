import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary

from models.t5.tokenization_t5_fast import T5TokenizerFast 
from models.pix2struct.image_processing_pix2struct import Pix2StructImageProcessor
from models.pix2struct.processing_pix2struct import Pix2StructProcessor
from models.pix2struct.modeling_pix2struct import Pix2StructForConditionalGeneration

from configs import matcha_base_train_Config
from datasets import SciCapPlusDataset
from utils import torch_fix_seed, train_one_epoch, test_one_epoch, freeze_pix2struct_parameters

config = matcha_base_train_Config()

if config.wandb == True:
    import wandb
    wandb.init(project="Pix2Struct_FT",
           name=config.project_name,
           config=config)
else:
    wandb = None

print("=== config data === \n", config)
for key, value in config.__dict__.items():
    print(f"{key}: {value}")
print("-------------------")


# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')

# check project name
save_dir = f'./run_result_{config.project_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print("train [info] : This project name is already running")
    raise RuntimeError("Duplicate project name. Program stopped.")


image_processor = Pix2StructImageProcessor.from_pretrained(config.pretrained_model_name)
tokenizer = T5TokenizerFast.from_pretrained(config.pretrained_model_name)
processor = Pix2StructProcessor(image_processor = image_processor,
                                tokenizer = tokenizer)
print("processor [info] : ", processor.image_processor.is_vqa)
processor.image_processor.is_vqa = False # vqaタスクの設定をvqa=TueからFalseに変更 >> これにより，"decoder_attention_mask", "decoder_input_ids"が追加される．
print("processor [info] : ", processor.image_processor.is_vqa)

model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(device)
print("train [info] : model.dtype : ", model.dtype)
freeze_pix2struct_parameters(model)
summary(model)

if config.is_DataParallel == True:
    model = nn.DataParallel(model)


optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), 
                            eps=1e-6, weight_decay=config.weight_decay)
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.t0)


train_dataset = SciCapPlusDataset(scicap_plus_dataset_path="/taiga/Datasets/scicap_plus",
                                  is_train = True,
                                  contains_subfigure = False,
                                  include_val = False,
                                  generate= False,
                                  template="Caption for this figure : ",
                                  )

test_dataset = SciCapPlusDataset(scicap_plus_dataset_path="/taiga/Datasets/scicap_plus",
                                 is_train = False,
                                 contains_subfigure = False,
                                 include_val = False,
                                 generate= False,
                                 template="Caption for this figure : ",
                                 )


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          )



for epoch in range(1, config.num_eposh + 1, 1):
    start_time = time.time()  # Record the start time of the epoch
    
    train_loss = train_one_epoch(model=model, 
                                 train_loader=train_loader, 
                                 processor=processor, 
                                 optimizer=optimizer, 
                                 config=config, 
                                 device=device,
                                 wandb=wandb
                                 )
    
    test_loss = test_one_epoch(model=model, 
                                test_loader=test_loader, 
                                processor=processor, 
                                config=config, 
                                device=device,
                                wandb=wandb
                                )
    
    lr_scheduler.step()
    
    end_time = time.time()  # Record the end time of the epoch
    epoch_time = end_time - start_time  # Calculate the time taken for the epoch
    
    if config.wandb:
        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "test_loss": test_loss,
                   })
    
    print(f"[epoch:{epoch}] train_loss:{train_loss} test_loss:{test_loss} time:{epoch_time:.2f} seconds")

    # Save the model every 5 epochs
    if epoch % config.save_param_time == 0:
        model_save_path = f"./run_result_{config.project_name}/run_result_{config.project_name}_epoch_{epoch}.pth"
        if config.is_DataParallel:
            torch.save(model.module.state_dict(), model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch} to {model_save_path}")
