"""
    Fixed-token variant of the packed LVR SFT dataset.
    Uses a fixed number of LVR tokens per region (max_lvr_tokens) instead of
    computing the count from bounding boxes. In this mode, LVR tokens are NOT
    extracted from the original image; the token positions are kept as-is.
"""

import copy
import os
import torch
import transformers
import ujson as json

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
)
from transformers import TrainingArguments

from .data_utils import get_image_info, llava_to_openai_lvr, pad_sequence
from .lvr_sft_dataset_packed import PackedDataset, PackedDataCollatorForSupervisedDatasetLVR

import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset


class IterableSupervisedDatasetLVRFixedToken(Dataset):
    """
    Packed dataset variant where each LVR region gets exactly max_lvr_tokens
    placeholder tokens instead of a bbox-derived count.
    """

    def __init__(
        self,
        data_path,
        image_folder,
        processor,
        data_args: DataArguments,
        ds_name,
        model_id,
        max_lvr_tokens: int,
        data_rank=0,
        data_world_size=1,
        distributed_mode=True,
        random_seed=None,
        latent_end_token=False,
    ):
        super().__init__()
        if isinstance(data_path, str):
            self.raw_data = json.load(open(data_path, "r"))
        else:
            self.raw_data = data_path

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.image_folder = image_folder
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.ds_name = ds_name
        self.fps = data_args.fps
        self.max_lvr_tokens = max_lvr_tokens

        self.data_world_size = data_world_size
        self.worker_id = None
        self.distributed_mode = distributed_mode
        self.worker_distributed = False
        self._state_dict = {}

        self.random_seed = None
        if random_seed:
            self.random_seed = random_seed
            self.rng = np.random.default_rng(seed=self.random_seed)
            self.rng.shuffle(self.raw_data)

        self.latent_end_token = latent_end_token

    def __len__(self):
        return len(self.raw_data)

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0
        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']
            self._state_dict.pop(self.worker_state_key)

        for i in range(start_idx, len(self.raw_data)):
            sources = self.raw_data[i]

            is_video = False
            processor = self.processor
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(
                    image_file,
                    self.image_min_pixel, self.image_max_pixel,
                    self.image_resized_w, self.image_resized_h,
                ))

            # In fixedToken mode: skip bbox → token-index extraction.
            # Use fixed_num_of_lvr_tokens so the text gets exactly max_lvr_tokens
            # LVR placeholder tokens per region, with no image-region extraction.
            sources = copy.deepcopy(llava_to_openai_lvr(
                sources['conversations'],
                is_video=is_video,
                lvr_token_idxs_list=None,
                latent_end_token=self.latent_end_token,
                fixed_num_of_lvr_tokens=self.max_lvr_tokens,
            ))

            all_input_ids = []
            all_labels = []
            all_pixel_values = []
            all_image_grid_thw = []

            if len(SYSTEM_MESSAGE) > 0:
                system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
                system_message_input_ids = processor.tokenizer(
                    system_message, add_special_tokens=False, return_tensors='pt'
                )['input_ids']
                system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)
                all_input_ids.append(system_message_input_ids.squeeze(0))
                all_labels.append(system_labels.squeeze(0))

            for _, j in enumerate(range(0, len(sources), 2)):
                user_input = sources[j]
                gpt_response = sources[j + 1]

                user_input_text = (
                    f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n"
                    f"{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
                    f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                )
                gpt_response_text = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

                if DEFAULT_IMAGE_TOKEN in user_input_text:
                    inputs = processor(
                        text=[user_input_text], images=images, videos=videos,
                        padding=False, do_resize=False, return_tensors='pt'
                    )
                    prompt_input_ids = inputs['input_ids']
                    all_pixel_values.append(inputs[pixel_key])
                    all_image_grid_thw.append(inputs[grid_key])
                else:
                    prompt_input_ids = processor.tokenizer(
                        user_input_text, add_special_tokens=False,
                        padding=False, return_tensors='pt'
                    )['input_ids']

                response_input_ids = processor.tokenizer(
                    gpt_response_text, add_special_tokens=False,
                    padding=False, return_tensors='pt'
                )['input_ids']

                input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                        response_input_ids.squeeze(0),
                    ],
                    dim=0,
                )
                all_input_ids.append(input_ids)
                all_labels.append(labels)

            input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
            labels = torch.cat(all_labels, dim=0).to(torch.long)
            attention_mask = (input_ids > -1000000).to(torch.long)

            # No image-region extraction in fixedToken mode
            lvr_tokens = []

            data_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                lvr_tokens=lvr_tokens,
            )

            if pixel_key and grid_key:
                pixel_values = torch.cat(all_pixel_values, dim=0)
                image_thw = torch.cat(all_image_grid_thw, dim=0)
                data_dict[pixel_key] = pixel_values
                data_dict[grid_key] = image_thw

            data_dict['input_lengths'] = torch.tensor([input_ids.size(0)])

            yield data_dict


def make_packed_supervised_data_module_lvr_fixedToken(
    model_id,
    processor,
    max_lvr_tokens: int,
    data_args: DataArguments,
    training_args: TrainingArguments,
    latent_end_token=False,
):
    """Make dataset and collator for packed SFT with fixed-count LVR tokens."""

    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()

    meta_data = json.load(open(data_args.data_path))

    datasets = []
    total_data_len = 0
    for meta in meta_data:
        iterable_sft_dataset = IterableSupervisedDatasetLVRFixedToken(
            data_path=meta['data_path'],
            image_folder=meta['image_folder'],
            ds_name=meta['ds_name'],
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            max_lvr_tokens=max_lvr_tokens,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=training_args.enable_data_packing,
            random_seed=data_args.random_seed,
            latent_end_token=latent_end_token,
        )
        datasets.append(iterable_sft_dataset)
        total_data_len += len(iterable_sft_dataset)

    packed_train_dataset = PackedDataset(
        tokenizer=processor.tokenizer,
        datasets=datasets,
        data_rank=data_rank,
        data_world_size=data_world_size,
        max_packed_tokens=training_args.max_packed_tokens,
        max_buffer_size=100,
        long_seq_threshold=training_args.long_seq_threshold,
        max_instance_per_batch=training_args.max_instance_per_batch,
    )

    data_collator = PackedDataCollatorForSupervisedDatasetLVR(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=packed_train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    ), total_data_len
