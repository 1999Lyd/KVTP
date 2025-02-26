#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.prune_rate_list = []
            self.add_frame_idx = False
            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def add_predictor(self, path):
        self.temp_tokenizer = AutoTokenizer.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2")
        
        if path == None:
            self.predictor = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to('cuda')
            self.siglip_model = self.predictor
        else:
            self.predictor = AutoModel.from_pretrained(path).to('cuda')
            self.siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to('cuda')
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.predictor.half()
    def get_predictor(self):
        return getattr(self, 'siglip_model', None)
    
    def activate_prune(self, prune_method):
        self.prune_method = prune_method
    
    def get_prune(self):
        prune = getattr(self, 'prune_method', None)
        return prune

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

import numpy as np

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def hook_attention(module, input, output):
    # Extract the attention matrix from the output
    attn_matrix = output[1]  # `output[1]` is `attn_output_weights`
    #print("Attention Matrix Shape:", attn_matrix.shape)
    # Optionally store it for later use
    module.attention_matrix = attn_matrix

def hook_key(module, input, output):
    # `input` is a tuple containing the query, key, and value tensors
    keys = input[1]  # `input[1]` corresponds to the key tensor
    #print("Key Tensor Shape:", keys.shape)
    module.key_tensor = keys

def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def add_predictor(self, path):
        return self.get_model().add_predictor(path)

    def activate_prune(self, prune_method):
        return self.get_model().activate_prune(prune_method)
    
    def add_frame_idx(self):
        return self.get_model().add_frame_idx

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images, use_cls = False):
        '''
        self.get_model().siglip_model.half()
        if use_cls:
            post_layernorm = self.get_model().siglip_model.vision_model.post_layernorm
            head = self.get_model().siglip_model.vision_model.head
            #probe = head.probe
            last_layer = self.get_model().siglip_model.vision_model.encoder.layers[-1]
        else:
            head = None
            last_layer = None
            post_layernorm = None
        '''
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features
    
    
    def tome(self, image_features, images, if_adaptive=False, reduction_ratio = 0.1, rate_list = None):
        '''
        version 24/03/2024 using the spacially smapled tokens to supplement the pruned tokens
        '''
        # token_indix_list = []
        # token_indix_dict = {}
        #rate_list = None
        #set hooks for extracting desired layer's k and q
        #hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        #hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)
        attention = self.get_model().siglip_model.vision_model.head.attention
        attention_hook_handle = attention.register_forward_hook(hook_attention)
        key_hook_handle = attention.register_forward_hook(hook_key)
        #hook_handle_k = self.get_model().siglip_model.vision_model.head.attention.k_proj.register_forward_hook(hook_k)
        #hook_handle_q = self.get_model().siglip_model.vision_model.head.attention.q_proj.register_forward_hook(hook_q)
        #forward pass
        image_forward_outs = self.get_model().siglip_model.vision_model(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        #cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        #image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape
        #extract desired layer's k and q and remove hooks; calculate attention
        #desired_layer_k = outputs["desired_k"]
        #desired_layer_q = outputs["desired_q"]
        attention_hook_handle.remove()
        key_hook_handle.remove()
        #hook_handle_k.remove()
        #hook_handle_q.remove()
        
        #attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        cls_attn = attention.attention_matrix
        desired_layer_k = attention.key_tensor
        #attn = F.softmax(attn, dim=-1)
        cls_attn = cls_attn.view(B, 1, 27, 27)  # Reshape to spatial dimensions (B, 1, 27, 27)
        cls_attn = F.avg_pool2d(cls_attn, kernel_size=2, stride=2)  # Average pooling with stride=2
        cls_attn = cls_attn.view(B, 169, 1).squeeze(-1) 
        desired_layer_k = desired_layer_k.view(B, -1, 27, 27)  # Reshape to spatial dimensions (B, 1, 27, 27)
        desired_layer_k = F.avg_pool2d(desired_layer_k, kernel_size=2, stride=2)  # Average pooling with stride=2
        desired_layer_k = desired_layer_k.view(B, 169, -1) 
        
        #cls_attn = attn[:, 0, 1:]  
        if rate_list is not None:
            new_frame_features_list = []
            original_indices_list = []

            for i in range(B):
                rate = rate_list[i]
                N = image_features.shape[1]
                r = int(N * (1 - rate))
                frame_features = image_features[i]
                frame_keys = desired_layer_k[i]

                original_indices = torch.arange(N, device=frame_features.device)

                while r > 0:
                    # Split tokens into two groups
                    even_idx = torch.arange(0, N, 2, device=frame_features.device)
                    odd_idx = torch.arange(1, N, 2, device=frame_features.device)

                    group1 = frame_features[even_idx]
                    group2 = frame_features[odd_idx]
                    keys1 = frame_keys[even_idx]
                    keys2 = frame_keys[odd_idx]

                    indices1 = original_indices[even_idx]
                    indices2 = original_indices[odd_idx]

                    # Compute cosine similarity between tokens in group2 and group1
                    similarity = F.cosine_similarity(keys2.unsqueeze(1), keys1.unsqueeze(0), dim=-1)

                    # Select top-k tokens from group2 based on similarity
                    max_sim, max_idx = similarity.max(dim=1)
                    k = min(r, len(group2))
                    top_k_sim, top_k_indices = torch.topk(max_sim, k=k, largest=True)
                    selected_indices = top_k_indices.tolist()

                    # Initialize a list to keep track of merging information
                    merge_map = [[] for _ in range(len(group1))]

                    # Determine merging tokens for each token in group1
                    for idx in selected_indices:
                        target_idx = max_idx[idx].item()
                        merge_map[target_idx].append(idx)

                    # Perform merging for group1 tokens
                    for g1_idx, merge_idxs in enumerate(merge_map):
                        if not merge_idxs:
                            continue

                        # Get the tokens to merge
                        merging_tokens = torch.cat([group1[g1_idx].unsqueeze(0), group2[merge_idxs]], dim=0)
                        merging_keys = torch.cat([keys1[g1_idx].unsqueeze(0), keys2[merge_idxs]], dim=0)

                        # Compute the new token and key
                        max_norm = torch.norm(merging_tokens, p=2, dim=-1).max()
                        group1[g1_idx] = (
                            merging_tokens.mean(dim=0).div(torch.norm(merging_tokens.mean(dim=0), p=2)) * max_norm
                        )
                        keys1[g1_idx] = (
                            merging_keys.mean(dim=0).div(torch.norm(merging_keys.mean(dim=0), p=2)) * max_norm
                        )

                    # Remove merged tokens from group2 and keys2
                    remaining_idxs = [idx for idx in range(len(group2)) if idx not in selected_indices]
                    group2 = group2[remaining_idxs]
                    keys2 = keys2[remaining_idxs]
                    indices2 = indices2[remaining_idxs]

                    # Update the frame tokens and keys
                    frame_features = torch.cat([group1, group2], dim=0)
                    frame_keys = torch.cat([keys1, keys2], dim=0)
                    original_indices = torch.cat([indices1, indices2], dim=0)

                    # Update r and N
                    r -= k
                    N = frame_features.shape[0]

                # Update lists with new frame features and original indices
                new_frame_features_list.append(frame_features)
                original_indices_list.append(original_indices)
                
            return new_frame_features_list, original_indices_list
        else:      
            new_frame_features_list = []
            original_indices_list = []

            for i in range(B):
                rate = reduction_ratio
                N = image_features.shape[1]
                r = int(N * (1 - rate))
                frame_features = image_features[i]
                frame_keys = desired_layer_k[i]

                original_indices = torch.arange(N, device=frame_features.device)

                while r > 0:
                    # Split tokens into two groups
                    even_idx = torch.arange(0, N, 2, device=frame_features.device)
                    odd_idx = torch.arange(1, N, 2, device=frame_features.device)
                    
                    group1 = frame_features[even_idx]
                    group2 = frame_features[odd_idx]
                    keys1 = frame_keys[even_idx]
                    keys2 = frame_keys[odd_idx]

                    indices1 = original_indices[even_idx]
                    indices2 = original_indices[odd_idx]

                    # Compute cosine similarity between tokens in group2 and group1
                    similarity = F.cosine_similarity(keys2.unsqueeze(1), keys1.unsqueeze(0), dim=-1)

                    # Select top-k tokens from group2 based on similarity
                    max_sim, max_idx = similarity.max(dim=1)
                    k = min(r, len(group2))
                    top_k_sim, top_k_indices = torch.topk(max_sim, k=k, largest=True)
                    selected_indices = top_k_indices.tolist()

                    # Initialize a list to keep track of merging information
                    merge_map = [[] for _ in range(len(group1))]

                    # Determine merging tokens for each token in group1
                    for idx in selected_indices:
                        target_idx = max_idx[idx].item()
                        merge_map[target_idx].append(idx)

                    # Perform merging for group1 tokens
                    for g1_idx, merge_idxs in enumerate(merge_map):
                        if not merge_idxs:
                            continue

                        # Get the tokens to merge
                        merging_tokens = torch.cat([group1[g1_idx].unsqueeze(0), group2[merge_idxs]], dim=0)
                        merging_keys = torch.cat([keys1[g1_idx].unsqueeze(0), keys2[merge_idxs]], dim=0)

                        # Compute the new token and key
                        max_norm = torch.norm(merging_tokens, p=2, dim=-1).max()
                        group1[g1_idx] = (
                            merging_tokens.mean(dim=0).div(torch.norm(merging_tokens.mean(dim=0), p=2)) * max_norm
                        )
                        keys1[g1_idx] = (
                            merging_keys.mean(dim=0).div(torch.norm(merging_keys.mean(dim=0), p=2)) * max_norm
                        )

                    # Remove merged tokens from group2 and keys2
                    remaining_idxs = [idx for idx in range(len(group2)) if idx not in selected_indices]
                    group2 = group2[remaining_idxs]
                    keys2 = keys2[remaining_idxs]
                    indices2 = indices2[remaining_idxs]

                    # Update the frame tokens and keys
                    frame_features = torch.cat([group1, group2], dim=0)
                    frame_keys = torch.cat([keys1, keys2], dim=0)
                    original_indices = torch.cat([indices1, indices2], dim=0)

                    # Update r and N
                    r -= k
                    N = frame_features.shape[0]

                # Update lists with new frame features and original indices
                new_frame_features_list.append(frame_features)
                original_indices_list.append(original_indices)
                
            return new_frame_features_list, original_indices_list
        
        
    def prumerge(self, image_features, images, if_adaptive=False, reduction_ratio = 0.2, rate_list = None):
        '''
        version 24/03/2024 using the spacially smapled tokens to supplement the pruned tokens
        '''

        attention = self.get_model().siglip_model.vision_model.head.attention
        attention_hook_handle = attention.register_forward_hook(hook_attention)
        key_hook_handle = attention.register_forward_hook(hook_key)
        
        #forward pass
        image_forward_outs = self.get_model().siglip_model.vision_model(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        #cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        
        B, N, C = image_features.shape
        
        attention_hook_handle.remove()
        key_hook_handle.remove()
       
        #attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        cls_attn = attention.attention_matrix
        desired_layer_k = attention.key_tensor
        #attn = F.softmax(attn, dim=-1)
        cls_attn = cls_attn.view(B, 1, 27, 27)  # Reshape to spatial dimensions (B, 1, 27, 27)
        cls_attn = F.avg_pool2d(cls_attn, kernel_size=2, stride=2)  # Average pooling with stride=2
        cls_attn = cls_attn.view(B, 169, 1).squeeze(-1) 
        desired_layer_k = desired_layer_k.view(B, -1, 27, 27)  # Reshape to spatial dimensions (B, 1, 27, 27)
        desired_layer_k = F.avg_pool2d(desired_layer_k, kernel_size=2, stride=2)  # Average pooling with stride=2
        desired_layer_k = desired_layer_k.view(B, 169, -1) 
        
        #cls_attn = attn[:, 0, 1:]  
        if rate_list is not None:
            new_frame_features_list = []
            original_indices_list = []
            for b in range(B):
                frame_features = image_features[b]
                reduction_ratio = rate_list[b]
                if reduction_ratio>1:
                    reduction_ratio=1
                _, idx = torch.topk(cls_attn[b], int(N*reduction_ratio), dim=0, largest=True)  # [left_tokens] , sorted=True
                index = idx.unsqueeze(-1).expand(-1, C)  # [left_tokens, C]
                key_index = idx.unsqueeze(-1).expand(-1, 1152)
                Key_wo_cls = desired_layer_k[b]  # [N, C]

                x_others = torch.gather(frame_features, dim=0, index=index)  # [left_tokens, C]
                x_others_attn = torch.gather(cls_attn[b], dim=0, index=idx)  
                Key_others = torch.gather(Key_wo_cls, dim=0, index=key_index)  # [left_tokens, C]
                compl = complement_idx(idx, N)  # [N-1-left_tokens]
                non_topk = torch.gather(frame_features, dim=0, index=compl.unsqueeze(-1).expand(-1, C))  # [B, N-1-left_tokens, C]
                non_topk_Key = torch.gather(Key_wo_cls, dim=0, index=compl.unsqueeze(-1).expand(-1, 1152))
                non_topk_attn = torch.gather(cls_attn[b], dim=0, index=compl)  # [B, N-1-left_tokens]

                Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
                non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

                left_tokens, C = x_others.size()
                updated_x_others = torch.zeros_like(x_others)

        #for b in range(B):
                for i in range(left_tokens):
                    key_others_norm = Key_others_norm[i,:].unsqueeze(0).unsqueeze(0)

                    before_i_Key = Key_others_norm[:i, :].unsqueeze(0)  
                    after_i_Key = Key_others_norm[i+1:, :].unsqueeze(0) 

                    before_i_x_others = x_others[:i, :].unsqueeze(0)  
                    after_i_x_others = x_others[i+1:, :].unsqueeze(0)   
                    rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[:,:].unsqueeze(0)], dim=1)   
                    before_i_x_others_attn = x_others_attn[:i].unsqueeze(0)  
                    after_i_x_others_attn = x_others_attn[i+1:].unsqueeze(0)  
                    rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[:].unsqueeze(0)], dim=1)  

                    rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[:,:].unsqueeze(0)], dim=1)
                    cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                    _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)


                    cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                    weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                    # update cluster centers
                    weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                    updated_center = weighted_avg + x_others[i, :]  
                    updated_x_others[i, :] = updated_center 
            

                #extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                #updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
                frame_features = updated_x_others
                new_frame_features_list.append(frame_features)
                original_indices_list.append(idx)
            return new_frame_features_list,original_indices_list

                
        else:      
            new_frame_features_list = []
            original_indices_list = []
            for b in range(B):
                frame_features = image_features[b]
                #reduction_ratio = rate_list[b]

                _, idx = torch.topk(cls_attn[b], int(N*reduction_ratio), dim=0, largest=True)  # [left_tokens] , sorted=True
                index = idx.unsqueeze(-1).expand(-1, C)  # [left_tokens, C]
                key_index = idx.unsqueeze(-1).expand(-1, 1152)
                Key_wo_cls = desired_layer_k[b]  # [N, C]

                x_others = torch.gather(frame_features, dim=0, index=index)  # [left_tokens, C]
                x_others_attn = torch.gather(cls_attn[b], dim=0, index=idx)  
                Key_others = torch.gather(Key_wo_cls, dim=0, index=key_index)  # [left_tokens, C]
                compl = complement_idx(idx, N)  # [N-1-left_tokens]
                non_topk = torch.gather(frame_features, dim=0, index=compl.unsqueeze(-1).expand(-1, C))  # [B, N-1-left_tokens, C]
                non_topk_Key = torch.gather(Key_wo_cls, dim=0, index=compl.unsqueeze(-1).expand(-1, 1152))
                non_topk_attn = torch.gather(cls_attn[b], dim=0, index=compl)  # [B, N-1-left_tokens]

                Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
                non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

                left_tokens, C = x_others.size()
                updated_x_others = torch.zeros_like(x_others)

        #for b in range(B):
                for i in range(left_tokens):
                    key_others_norm = Key_others_norm[i,:].unsqueeze(0).unsqueeze(0)

                    before_i_Key = Key_others_norm[:i, :].unsqueeze(0)  
                    after_i_Key = Key_others_norm[i+1:, :].unsqueeze(0) 

                    before_i_x_others = x_others[:i, :].unsqueeze(0)  
                    after_i_x_others = x_others[i+1:, :].unsqueeze(0)   
                    rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[:,:].unsqueeze(0)], dim=1)   
                    before_i_x_others_attn = x_others_attn[:i].unsqueeze(0)  
                    after_i_x_others_attn = x_others_attn[i+1:].unsqueeze(0)  
                    rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[:].unsqueeze(0)], dim=1)  

                    rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[:,:].unsqueeze(0)], dim=1)
                    cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                    _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)


                    cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                    weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                    # update cluster centers
                    weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                    updated_center = weighted_avg + x_others[i, :]  
                    updated_x_others[i, :] = updated_center 
            

                #extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                #updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
                frame_features = updated_x_others
                new_frame_features_list.append(frame_features)
                original_indices_list.append(idx)
            return new_frame_features_list,original_indices_list
        
    def add_token_per_grid_prune(self, image_feature, images, rate_list, prune_method):
        
        num_frames = image_feature.shape[0]  
        feature_dim = image_feature.shape[-1]
        
        if prune_method == 'prumerge':
            pruned_features, pruned_indices = self.prumerge(image_feature, images, if_adaptive=False, reduction_ratio = 1/10, rate_list = rate_list)
        elif prune_method == 'tome': 
            pruned_features, pruned_indices = self.tome(image_feature, images, if_adaptive=False, reduction_ratio = 1/10, rate_list = rate_list)
        # Step 2: Initialize lists to hold modified tokens and the final pruned indices per frame
        modified_features = []

        for frame_idx in range(num_frames):
            # Get the pruned tokens and indices for this frame
            frame_tokens = pruned_features[frame_idx]  # Shape: (pruned_num_tokens, feature_dim)
         
            frame_indices = pruned_indices[frame_idx]  # Indices of the pruned tokens
            if len(frame_indices) > 0 :
            # Step 3: Calculate the row positions of the pruned tokens
            # Assuming original image tokens form a grid of size (resize_h x resize_h) before pruning
                num_original_tokens = image_feature.shape[1]
                resize_h = int(math.sqrt(num_original_tokens))  # Assuming square grid before pruning
                pruned_rows = frame_indices // resize_h  # Calculate the row positions of pruned tokens
                
                # Step 4: Insert special token between tokens that are from different rows
                special_token = self.model.image_newline.to(image_feature.device)
                modified_tokens = []
                
                for i in range(frame_tokens.shape[0] - 1):
                    modified_tokens.append(frame_tokens[i])
                    
                    # Insert special token between tokens from different rows
                    if pruned_rows[i] != pruned_rows[i + 1]:
                        modified_tokens.append(special_token)

                modified_tokens.append(frame_tokens[-1])  # Append the last token

                # Step 5: Stack the modified tokens into a tensor
                modified_tokens = torch.stack(modified_tokens).view(-1, feature_dim)
                modified_features.append(modified_tokens)
            else:
                continue
        
        # Step 6: Concatenate all modified frames back into a single tensor
        modified_features = torch.cat(modified_features,dim=0)
        
        # Step 7: Handle the 'add_faster_video' flag for further processing, if needed
        if getattr(self.config, "add_faster_video", False):
            modified_features = modified_features.permute(0, 2, 1, 3).contiguous()
            modified_features = modified_features.view(num_frames, -1, feature_dim)
            return modified_features

        # Return the modified features and the indices of pruned tokens
        return modified_features
    
    def add_token_per_grid_random(self, image_feature, rate_list):
        #rate_list=None
        resize_h = int(math.sqrt(image_feature.shape[1]))  # since there are 14 additional tokens per frame
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]
        
        # Reshape the image features to the grid format (excluding special tokens for now)
        image_feature = image_feature.view(num_frames, resize_h, resize_h, -1)

        # Initialize lists to hold modified tokens and sampled indices per frame
        modified_features = []
        indices_per_frame = []

        for frame_idx in range(num_frames):
            # Flatten the tokens for sampling (excluding special tokens for now)
            frame_tokens = image_feature[frame_idx].view(-1, feature_dim)  # Shape: (196, feature_dim)
            
            # Randomly sample 20% of the tokens
            num_tokens_per_frame = frame_tokens.shape[0]  # 196 tokens per frame
            if rate_list is not None:
                num_sampled_tokens = int(num_tokens_per_frame * rate_list[frame_idx])
            else:
                if frame_idx<num_frames/2:
                    num_sampled_tokens = int(num_tokens_per_frame * 0.1)  # 15% of tokens
                else:
                    num_sampled_tokens = int(num_tokens_per_frame * 0.1)
               
                
            num_sampled_tokens = min(num_sampled_tokens,num_tokens_per_frame)
            sampled_indices = torch.randperm(num_tokens_per_frame)[:num_sampled_tokens]
            
            # Sort the sampled indices to maintain their original order
            sampled_indices = torch.sort(sampled_indices).values
            indices_per_frame.append(sampled_indices)  # Store the sampled indices
            
            # Get the sampled tokens and restore row information
            sampled_tokens = frame_tokens[sampled_indices]
            sampled_rows = sampled_indices // resize_h  # Get row numbers of sampled tokens
            
            # Insert a special token between tokens from different rows
            special_token = self.model.image_newline.to(image_feature.device)
            modified_tokens = []
            if sampled_tokens.shape[0] > 0:
                for i in range(sampled_tokens.shape[0] - 1):
                    modified_tokens.append(sampled_tokens[i])
                    
                    # Add a special token only if the next token belongs to a different row
                    if sampled_rows[i] != sampled_rows[i + 1]:
                        modified_tokens.append(special_token)
                
                modified_tokens.append(sampled_tokens[-1])  # Append the last token without adding special token at the end
                modified_tokens.append(special_token)
            else:
                modified_tokens.append(special_token)
            # Stack the modified tokens back into a tensor
            modified_tokens = torch.stack(modified_tokens).view(-1, feature_dim)
            modified_features.append(modified_tokens)
      
        # Concatenate all modified frames
        modified_features = torch.cat(modified_features,dim=0)

        # If the faster video option is enabled, do further processing
        if getattr(self.config, "add_faster_video", False):
            modified_features = modified_features.permute(0, 2, 1, 3).contiguous()
            modified_features = modified_features.view(num_frames, -1, feature_dim)
            return modified_features
        
        return modified_features

    def add_token_per_grid(self, image_feature):
    
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
 
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
  
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
    
        if self.add_frame_idx():
            image_feature = image_feature.view(num_frames, resize_h, resize_h+1, -1)
            image_feature = image_feature.flatten(1, 2)
            i_feature = []
            for i, frame_feature in enumerate(image_feature):
                frame_idx = "frame {}:".format(i+1)
                frame_idx_id = torch.tensor(self.get_model().temp_tokenizer(frame_idx).input_ids).to(frame_feature.device)
                frame_idx_embed = self.get_model().embed_tokens(frame_idx_id)
             
                frame_feature = torch.cat((frame_feature, frame_idx_embed.to(frame_feature.device)),dim=0)
    
                i_feature.append(frame_feature)
            image_feature = torch.cat(i_feature, dim=0)
            return image_feature

        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature
    
    def max_pooling_inference(self, logits, clip_size=8):
        """
        Applies max pooling on logits for a single video.

        Args:
            logits: Tensor of shape (L,), where L is the number of frames.
            clip_size: Number of frames per clip (default = 8).

        Returns:
            pooled_logits: Tensor of shape (L,), where each frame's score is max-pooled.
        """
        num_frames = logits.shape[0]
        pooled_logits = torch.zeros_like(logits)

        num_clips = (num_frames + clip_size - 1) // clip_size  # Compute number of clips

        for i in range(num_clips):
            start_idx = i * clip_size
            end_idx = min(start_idx + clip_size, num_frames)

            # Extract logits for the current clip
            clip_logits = logits[start_idx:end_idx]

            # Max pooling
            pooled_value = clip_logits.max()

            # Assign max-pooled value to all frames in the clip
            pooled_logits[start_idx:end_idx] = pooled_value

        return pooled_logits
    def softmax_pooling_inference(logits, clip_size=8):
        """
        Applies softmax pooling on logits for a single video.

        Args:
            logits: Tensor of shape (L,), where L is the number of frames.
            clip_size: Number of frames per clip (default = 8).

        Returns:
            pooled_logits: Tensor of shape (L,), where each frame's score is pooled.
        """
        num_frames = logits.shape[0]
        pooled_logits = torch.zeros_like(logits)

        num_clips = (num_frames + clip_size - 1) // clip_size  # Compute number of clips

        for i in range(num_clips):
            start_idx = i * clip_size
            end_idx = min(start_idx + clip_size, num_frames)

            # Extract logits for the current clip
            clip_logits = logits[start_idx:end_idx]

            # Apply softmax pooling
            weights = torch.softmax(clip_logits, dim=0)  # Compute weights
            pooled_value = torch.sum(weights * clip_logits)  # Weighted sum

            # Assign pooled value to all frames in the clip
            pooled_logits[start_idx:end_idx] = pooled_value

        return pooled_logits
    
    def chunk_input_ids(self, input_ids, max_size=64):
        """
        Chunk input_ids into smaller tensors with a maximum size along the last dimension.

        Args:
            input_ids (torch.Tensor): Tensor of shape (1, N).
            max_size (int): Maximum size of each chunk along the last dimension.

        Returns:
            List[torch.Tensor]: List of chunks with each tensor having shape (1, chunk_size).
        """
        # Flatten the input tensor to ensure a single dimension
        input_ids = input_ids.flatten()
        # Chunk the tensor into pieces of max_size
        chunks = torch.split(input_ids, max_size)
        # Add batch dimension back (1,)
        chunks = [chunk.unsqueeze(0) for chunk in chunks]
        return chunks
    
    def calculate_relevance(self, q_ids, images, temperature=2, base_rate=0.2, if_hard=False):
        relevance_score_list = []
        self.get_model().siglip_model.half()
           
        inputs = self.get_model().siglip_processor(text=q_ids, return_tensors="pt")
        chunked_ids = self.chunk_input_ids(inputs['input_ids'], max_size=64)
        for q_id in chunked_ids:
            
            inputs['input_ids'] = q_id.to('cuda')
            inputs['pixel_values'] = images.to('cuda')
            with torch.no_grad():
                outputs = self.get_model().predictor(**inputs)
    
            relevance_score = outputs.logits_per_text
            relevance_score_list.append(relevance_score)
            
        relevance_scores = torch.cat(relevance_score_list,dim=0)
      
        if relevance_scores.shape[0] > 1:
            relevance_score,_ = torch.max(relevance_scores, dim=0)
        else:
            relevance_score = relevance_scores.squeeze(0)
        
        normalize = True
        if normalize:
            relevance_score = relevance_score - relevance_score.mean(dim=-1, keepdim=True)
        pooling_strategy = None
        if pooling_strategy == "max":
            relevance_score = self.max_pooling_inference(relevance_score)
        elif pooling_strategy == "softmax":
            relevance_score = self.softmax_pooling_inference(relevance_score)
        seq_len = len(relevance_score)
        if if_hard:
            k = int(0.1 * len(relevance_score))  
            threshold = torch.topk(relevance_score, k).values.min()  
            rate_list = (relevance_score >= threshold).float()
        rate_list = F.softmax(relevance_score/temperature)*seq_len*base_rate
       
        return rate_list

    def update_attention_mask(self, attention_mask, input_ids, labels):
        # Create new masks based on the conditions
        input_ids_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Find the index of the first occurrence of -200 in each sequence
        first_neg200_idx = (input_ids == -200).int().argmax(dim=1)
        
        # Mask the part on the left side of the first -200 in each sequence
        for i in range(input_ids.size(0)):
            if first_neg200_idx[i] > 0:
                input_ids_mask[i, :first_neg200_idx[i]+1] = False
        
        labels_mask = labels == -100
        
        # Combine the new masks with the original attention mask
        if labels==None:
            if attention_mask!=None:
                new_attention_mask = attention_mask & input_ids_mask
            else:
                new_attention_mask = input_ids_mask
            
        else:

            new_attention_mask = attention_mask & input_ids_mask & labels_mask
      
        new_attention_mask[:,-5:] = 0
        return new_attention_mask

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, rate_list=None, debaised_context=None, use_attn=True, if_hard=False):
        vision_tower = self.get_vision_tower()
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        router_attention_mask = self.update_attention_mask(attention_mask, input_ids, labels)

        new_input_ids = torch.where(input_ids == -200, torch.ones_like(input_ids), input_ids[0])

        text_embed = self.get_model().embed_tokens(new_input_ids).squeeze(0)
       
       
        
        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)
            if self.get_model().get_predictor():

                rate_list = self.calculate_relevance(debaised_context,concat_images,if_hard=if_hard)
               
            else:
                rate_list = rate_list
                
            
            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
        
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
        
                    image_features.append(self.get_2dPool(image_feat))
                   
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            #image_feature = self.add_token_per_grid(image_feature)
                            prune_method = self.get_model().get_prune()
                            if prune_method == 'prumerge' or prune_method == 'tome':
                                image_feature = self.add_token_per_grid_prune(image_feature, concat_images, rate_list, prune_method)
                                
                            elif prune_method == 'random':
                                image_feature = self.add_token_per_grid_random(image_feature,rate_list)

                            else:
                                image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)
                      
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
      
        start_idx = 14
        end_idx = 14+image_features[0].shape[0]
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels,(start_idx,end_idx)

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
