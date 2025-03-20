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
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from PIL import Image
import io
import pyarrow.parquet as pq
import random
import bisect
import pyarrow.fs as fs

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


class MyParquetDataset(Dataset):
    def __init__(self, root_dir, tokenizer=None, size=512,
                 text_encoder_architecture='CLIP', norm=False):
        random.seed(23)

        self.root_dir = root_dir
        self.dataset_receipt = {'MSCOCO_part1': {'total_num': 6212, 'ratio':1}, 'MSCOCO_part2': {'total_num': 6212, 'ratio':1}}

        self.tokenizer = tokenizer
        self.size = size
        self.text_encoder_architecture = text_encoder_architecture
        self.norm = norm

        self.hdfs = fs.HadoopFileSystem(host="", port=0000) # TODO: change to your own HDFS host and port
        self._init_mixed_parquet_dir_list()

        self.file_metadata = []
        self.cumulative_sizes = [0]
        total = 0
        for path in self.parquet_files:
            try:
                with pq.ParquetFile(path, filesystem=self.hdfs) as pf:
                    num_rows = pf.metadata.num_rows
                    self.file_metadata.append({
                        'path': path,
                        'num_rows': num_rows,
                        'global_offset': total
                    })
                    total += num_rows
                    self.cumulative_sizes.append(total)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue

        # init cache
        self.current_file = None
        self.cached_data = None
        self.cached_file_index = -1

    def _init_mixed_parquet_dir_list(self):
        print('Loading parquet files, please be patient...')
        self.parquet_files = []

        for key, value in self.dataset_receipt.items():
            # Generate a list of standard Parquet file paths, lazy load
            hdfs_path = os.path.join(self.root_dir, key)

            num = value['total_num']
            sampled_list = random.sample(
                [f"{hdfs_path}/train-{idx:05d}-of-{num:05d}.parquet" for idx in range(num)],
                k=int(num * value['ratio'])
            )
            self.parquet_files += sampled_list

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _locate_file(self, global_idx):
        # Use binary search to quickly locate files
        file_index = bisect.bisect_right(self.cumulative_sizes, global_idx) - 1
        if file_index < 0 or file_index >= len(self.file_metadata):
            raise IndexError(f"Index {global_idx} out of range")

        file_info = self.file_metadata[file_index]
        local_idx = global_idx - file_info['global_offset']
        return file_index, local_idx

    def _load_file(self, file_index):
        """Load Parquet files into cache on demand"""
        if self.cached_file_index != file_index:
            file_info = self.file_metadata[file_index]
            try:
                table = pq.read_table(file_info['path'], filesystem=self.hdfs)
                self.cached_data = table.to_pydict()
                self.cached_file_index = file_index
            except Exception as e:
                print(f"Error loading {file_info['path']}: {str(e)}")
                raise

    def __getitem__(self, idx):
        file_index, local_idx = self._locate_file(idx)
        self._load_file(file_index)
        sample = {k: v[local_idx] for k, v in self.cached_data.items()}

        # cprint(sample.keys(), 'red')
        generated_caption, image_path = sample['task2'], sample['image']  # only suitable for my data
        instance_image = Image.open(io.BytesIO(image_path['bytes']))

        # if instance_image.width < self.size or instance_image.height < self.size:
        #     raise ValueError(f"Image at {image_path} is too small")

        rv = process_image(instance_image, self.size, self.norm)

        if isinstance(self.tokenizer, list):
            _tmp_ = tokenize_prompt(self.tokenizer, generated_caption, self.text_encoder_architecture)
            rv["prompt_input_ids"] = [_tmp_[0][0], _tmp_[1][0]]
        else:
            rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, generated_caption, self.text_encoder_architecture)[
                0]

        return rv

class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_key,
        prompt_key,
        prompt_prefix=None,
        size=512,
        text_encoder_architecture='CLIP',
    ):
        self.size = size
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.prompt_prefix = prompt_prefix
        self.text_encoder_architecture = text_encoder_architecture

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]

        rv = process_image(item[self.image_key], self.size)

        prompt = item[self.prompt_key]

        if self.prompt_prefix is not None:
            prompt = self.prompt_prefix + prompt

        if isinstance(self.tokenizer, list):
            _tmp_ = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)
            rv["prompt_input_ids"] = [_tmp_[0][0],_tmp_[1][0]]
        else:
            rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)[0]

        return rv