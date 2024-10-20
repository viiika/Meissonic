# Copyright 2024 The HuggingFace Team and The MeissonFlow Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageOps import exif_transpose

import numpy as np
from PIL import Image
import io

import pandas as pd
import pyarrow.parquet as pq
import re


@torch.no_grad()
def tokenize_prompt(tokenizer, prompt, text_encoder_architecture='open_clip'): # only support open_clip and CLIP
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        return tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).input_ids
    elif text_encoder_architecture == 'CLIP_T5_base': # we have two tokenizers, 1st for CLIP, 2nd for T5
        input_ids = []
        input_ids.append(tokenizer[0](
            prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).input_ids)
        input_ids.append(tokenizer[1](
            prompt,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).input_ids)
        return input_ids
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")

def encode_prompt(text_encoder, input_ids, text_encoder_architecture='open_clip'):  # only support open_clip and CLIP
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        outputs = text_encoder(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        cond_embeds = outputs[0]
        return encoder_hidden_states, cond_embeds

    elif text_encoder_architecture == 'CLIP_T5_base':
        outputs_clip = text_encoder[0](input_ids=input_ids[0], return_dict=True, output_hidden_states=True)
        outputs_t5 = text_encoder[1](input_ids=input_ids[1], decoder_input_ids=torch.zeros_like(input_ids[1]),
                               return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs_t5.encoder_hidden_states[-2]
        cond_embeds = outputs_clip[0]
        return encoder_hidden_states, cond_embeds
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")


def process_image(image, size, Norm=False, hps_score = 6.0): 
    image = exif_transpose(image)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    orig_height = image.height
    orig_width = image.width

    image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(image)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(size, size))
    image = transforms.functional.crop(image, c_top, c_left, size, size)
    image = transforms.ToTensor()(image)

    if Norm:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)

    micro_conds = torch.tensor(
        [orig_width, orig_height, c_top, c_left, hps_score],
    )

    return {"image": image, "micro_conds": micro_conds}



class MyParquetDataset(Dataset): # Note this class may not work due to remove some sensitive code
    def __init__(self, parquet_dir, tokenizer, size=1024, text_encoder_architecture='open_clip', norm=False):
        self.size = size
        self.tokenizer = tokenizer
        self.parquet_dir = parquet_dir 

        print('Loading', parquet_dir,', please be patient, will cost some mins')
        if os.path.exists(parquet_dir):
            if os.path.isfile(parquet_dir):
                self.parquet_file = pq.ParquetFile(parquet_dir)
                self.row_group_tables = []
                for i in range(self.parquet_file.metadata.num_rows//1000000 + 1):
                    self.row_group_tables.append(self.parquet_file.read_row_group(i, use_threads=True))
            else:
                raise ValueError(f"Invalid path: {parquet_dir}")
        else:
            raise FileNotFoundError(f"Path does not exist: {parquet_dir}")
        print('Loaded')
        
        self.length = self._calculate_dataset_length()
        print('The length of traning dataset is ', self.length) 

        self.text_encoder_architecture = text_encoder_architecture
        self.norm = norm

    def _calculate_dataset_length(self):
        if os.path.isdir(self.parquet_dir):
            self.num_files = len(self.files)
            if self.num_files == 0:
                return 0

            try:
                self.rows_per_file = self.files[0].metadata.num_rows
                self.last_file_rows = self.files[-1].metadata.num_rows             
            except AttributeError:
                self.rows_per_file = 0
                self.last_file_rows = 0

            if self.num_files > 1:
                self.total_rows = (self.num_files - 1) * self.rows_per_file + self.last_file_rows
            else:
                self.total_rows = self.last_file_rows  

            return self.total_rows
        else:
            return self.parquet_file.metadata.num_rows

    def __len__(self):
        return self.length
    
    def get_file_index(piece):
        match = re.search(r'part_(\d+)\.parquet', piece)
        return int(match.group(1)) if match else 0
    
    def _extract_data(self, index):
        rows_per_file = 1000000
        
        file_index = index // rows_per_file
        row_index = index % rows_per_file
       
        return self.row_group_tables[file_index].slice(row_index, 1).column('caption')[0].as_py(), self.row_group_tables[file_index].slice(row_index, 1).column('image_path')[0].as_py(), self.row_group_tables[file_index].slice(row_index, 1).column('image_type')[0].as_py()
                
                
    def __getitem__(self, index):
      
        index = index % self.length
       
        generated_caption, image_path, image_type = self._extract_data(index)
       
        while True:
            try:
                if image_type == 'path':
                    assert f"You need to rewrite the code to load image from path"
                    res = self._load_image_from_oss(image_path) 
                    
                    imgbyte = res.read()
                    tempBuff = io.BytesIO()
                    tempBuff.write(imgbyte)
                    tempBuff.seek(0)
                    instance_image = Image.open(tempBuff)
                else: # image_type == 'local':
                    instance_image = Image.open(image_path)
               

                if instance_image.width < self.size or instance_image.height < self.size:
                    raise ValueError(f"Image at {image_path} is too small")

                rv = process_image(instance_image, self.size, self.norm)

    
                if self.text_encoder_architecture == 'CLIP_T5_base': # Not support now
                    _tmp_ = tokenize_prompt(self.tokenizer, generated_caption, self.text_encoder_architecture)
                    rv["prompt_input_ids"] = [_tmp_[0][0],_tmp_[1][0]]
                else:
                    rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, generated_caption, self.text_encoder_architecture)[0]
                
                return rv
            except:
                index += 1
                index = index % self.length
               
                generated_caption, image_path, image_type = self._extract_data(index)
              

class PickaPicV2Dataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        text_encoder_architecture='CLIP',
        norm=False,
    ):
        from glob import glob
        import os
        self.size = size
        self.tokenizer = tokenizer
        self.text_encoder_architecture = text_encoder_architecture
        self.norm = norm

        name_list = os.listdir(data_root)
        # self.instance_images_path = [os.path.join(data_root, name) for name in name_list if os.path.isfile(os.path.join(data_root, name)) and name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        lowercase_paths = [file for ext in extensions for file in glob(os.path.join(data_root, f"*{ext}"), recursive=True)]
        uppercase_paths = [file for ext in extensions for file in glob(os.path.join(data_root, f"*{ext.upper()}"), recursive=True)]
        self.instance_images_path = lowercase_paths + uppercase_paths

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        while True:
            try:
                image_path = self.instance_images_path[index % len(self.instance_images_path)]
                instance_image = Image.open(image_path)
                break
            except Exception as e:
                print(f"Error reading image at {image_path}: {e}")
                index = (index + 1)  
        
        rv = process_image(instance_image, self.size, self.norm)

        prompt_txt_path = image_path.replace("png","txt")
        with open(prompt_txt_path, 'r') as file:
            prompt = file.read()
        if self.text_encoder_architecture == 'CLIP_T5_base':  # Not support now
            _tmp_ = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)
            rv["prompt_input_ids"] = [_tmp_[0][0],_tmp_[1][0]]
        else:
            rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)[0]
        return rv


class MSCOCO600KDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        read_code=False,
        text_encoder_architecture='CLIP',
        norm=False,
    ):
        from glob import glob
        self.size = size
        self.tokenizer = tokenizer
        self.instance_images_path = glob(os.path.join(data_root,"**.jpg"))
        self.read_code = read_code
        self.text_encoder_architecture = text_encoder_architecture
        self.norm = norm

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        while True:
            try:
                image_path = self.instance_images_path[index % len(self.instance_images_path)]
                if self.read_code is not True:
                    instance_image = Image.open(image_path)
                    if instance_image.width < self.size or instance_image.height < self.size:
                        raise ValueError(f"Image at {image_path} is too small")
                if self.read_code:
                    rv = {}
                    micro_conds = torch.tensor(
                        [512, 512, 0, 0, 6.0],
                    )
                    rv["micro_conds"]=micro_conds
                else:
                    rv = process_image(instance_image, self.size, self.norm)
                base_name = image_path.split("/")[-1].split(".")[0]
                prompt_txt_path = image_path.replace("jpg","txt")
                if self.read_code:
                    code_path = image_path.replace("imgs","code_128x128").replace("jpg","npy")
                    code = np.load(code_path)
                    rv["code"] = torch.from_numpy(code)
                    
                with open(prompt_txt_path, 'r') as file:
                    prompt = file.read()
                rv["base_name"] = base_name
                if self.text_encoder_architecture == 'CLIP_T5_base': # Not support now
                    _tmp_ = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)
                    rv["prompt_input_ids"] = [_tmp_[0][0],_tmp_[1][0]]
                else:
                    rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)[0]
                return rv
            except:
                # print("failed read image file ", key)
                index += 1
                

class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_key,
        prompt_key,
        prompt_prefix=None,
        size=512,
    ):
        self.size = size
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.prompt_prefix = prompt_prefix

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]

        rv = process_image(item[self.image_key], self.size)

        prompt = item[self.prompt_key]

        if self.prompt_prefix is not None:
            prompt = self.prompt_prefix + prompt

        rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt)[0]

        return rv


