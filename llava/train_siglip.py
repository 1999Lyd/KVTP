import pandas as pd
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import os
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
import ast
from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
import deepspeed
from transformers.models.bart.modeling_bart import BartAttention
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer


BartAttention.use_memory_efficient_attention = True


class MultiLabelDataset(Dataset):
  def __init__(self, df):
    self.df = df
    self.max_frames_num = 128
    self.fps = 1
    self.force_sample = False
    self._image_processor = SigLipImageProcessor()
    self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

  def load_video(self, video_path, max_frames_num, fps, force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()

        return spare_frames, frame_time, video_time
  
  def chunk_input_ids(self, input_ids, max_size=64):
      
      # Flatten the input tensor to ensure a single dimension
      input_ids = input_ids.flatten()
      # Chunk the tensor into pieces of max_size
      chunks = torch.split(input_ids, max_size)
      # Add batch dimension back (1,)
      chunks = [chunk.unsqueeze(0) for chunk in chunks]
      return chunks
  def __getitem__(self, idx):
      item = self.df.iloc[idx]
      # get image
      video, frame_time, video_time = self.load_video(item.iloc[0], self.max_frames_num, self.fps, force_sample=self.force_sample)
      video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
      # prepare image for the model
      query = item.iloc[1]
      input_ids = self.siglip_processor(text=query , return_tensors="pt")['input_ids']
      chunked_ids = self.chunk_input_ids(input_ids, max_size=64)
      
      # get labels
      labels = item.iloc[2]
      
      labels = ast.literal_eval(labels)
      assert isinstance(labels, list)
      labels = torch.tensor(labels)
      # turn into PyTorch tensor
      #labels = torch.from_numpy(labels)

      return {"pixel_values": video, "input_ids": chunked_ids, "labels": labels}

  def __len__(self):
    return len(self.df)


class DataCollator:
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        input_ids = [f["input_ids"] for f in features]  # Keep as list for processing
        labels = torch.stack([f["labels"] for f in features])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}


class CustomModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        #self.loss_fn = nn.MSELoss() # Use your preferred loss function here
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.tau_local_prime = nn.Parameter(torch.ones(1) * 0.0) 
        self.tau_global_prime = nn.Parameter(torch.ones(1) * 0.0)
        
    @property
    def config(self):
        # Forward the config attribute from the original Hugging Face model
        return self.model.config
    
    def save_pretrained(self, save_directory, **kwargs):
        # Forward the call to the Hugging Face model
        self.model.save_pretrained(save_directory, **kwargs)
    
    def max_pooling_fixed_clips(self, logits):
        
        batch_size, num_frames = logits.shape
        clip_size = 8
        num_clips = (num_frames + clip_size - 1) // clip_size  # Number of clips per video

        clip_logits_list = []  # Store clip-level pooled logits for variance computation

        # Process each clip
        for b in range(batch_size):
            clip_logits_per_video = []

            for i in range(num_clips):
                start_idx = i * clip_size
                end_idx = min(start_idx + clip_size, num_frames)  # Handle last clip

                # Extract logits for the current clip
                clip_logits = logits[b, start_idx:end_idx]

                
                pooled_value = torch.max(clip_logits)  # Softmax-weighted sum
                
                # Store clip-level logits
                clip_logits_per_video.append(pooled_value)

            # Convert list to tensor
            clip_logits_tensor = torch.stack(clip_logits_per_video)  # Shape (num_clips,)

            # Compute variance if multiple clips exist
            if num_clips > 1:
                clip_variance_loss = -torch.var(clip_logits_tensor, dim=0).mean()  # Variance across clips
            else:
                clip_variance_loss = torch.tensor(0.0, device=logits.device)

            clip_logits_list.append(clip_logits_tensor)

        # Stack to form final logits tensor
        pooled_logits = torch.stack(clip_logits_list)  # Shape (B, num_clips)

        # Compute mean variance across batch
       

        return pooled_logits, clip_variance_loss

    def softmax_pooling_fixed_clips(self, logits):
        """
        Performs softmax pooling on logits for clips of fixed size (8 frames per clip),
        except for the last clip which may have fewer than 8 frames.

        Computes variance of clip-level pooled logits to encourage diversity.

        Args:
            logits: Tensor of shape (B, L), where L is the number of frames.

        Returns:
            pooled_logits: Tensor of shape (B, num_clips), where each clip has one pooled value.
            clip_variance_loss: Scalar tensor representing the variance loss across clips.
        """
        batch_size, num_frames = logits.shape
        clip_size = 8
        num_clips = (num_frames + clip_size - 1) // clip_size  # Number of clips per video

        clip_logits_list = []  # Store clip-level pooled logits for variance computation

        # Process each clip
        for b in range(batch_size):
            clip_logits_per_video = []

            for i in range(num_clips):
                start_idx = i * clip_size
                end_idx = min(start_idx + clip_size, num_frames)  # Handle last clip

                # Extract logits for the current clip
                clip_logits = logits[b, start_idx:end_idx]

                # Apply softmax pooling
                weights = torch.softmax(clip_logits, dim=0)  # Compute weights
                pooled_value = torch.sum(weights * clip_logits)  # Softmax-weighted sum
                
                # Store clip-level logits
                clip_logits_per_video.append(pooled_value)

            # Convert list to tensor
            clip_logits_tensor = torch.stack(clip_logits_per_video)  # Shape (num_clips,)

            # Compute variance if multiple clips exist
            if num_clips > 1:
                clip_variance_loss = -torch.var(clip_logits_tensor, dim=0).mean()  # Variance across clips
            else:
                clip_variance_loss = torch.tensor(0.0, device=logits.device)

            clip_logits_list.append(clip_logits_tensor)

        # Stack to form final logits tensor
        pooled_logits = torch.stack(clip_logits_list)  # Shape (B, num_clips)

        # Compute mean variance across batch
       

        return pooled_logits, clip_variance_loss

    def context_fusion_head(self, frame_embeddings, clip_size=8):

        N_frames, embed_dim = frame_embeddings.shape
        clips = torch.split(frame_embeddings, clip_size)  # Split into clips of size clip_size
        num_clips = len(clips)
        tau_local = torch.exp(self.tau_local_prime)
        tau_global = torch.exp(self.tau_global_prime)
        # Compute local context fused embeddings
        local_fused_embeddings = []
        for clip in clips:
            clip_size_actual = clip.shape[0]  # Handle the last clip if it has fewer frames
            normalized_clip = clip / (clip.norm(dim=-1, keepdim=True) + 1e-6)  # Normalize keys
            for frame in clip:
                # Cross-attention between the frame and all frames in the clip
                normalized_frame = frame / (frame.norm(dim=-1, keepdim=True) + 1e-6)  # Normalize query
                attention_weights = torch.softmax(
                    torch.matmul(normalized_frame.unsqueeze(0), normalized_clip.transpose(0, 1)) /
                    (tau_local * embed_dim ** 0.5),
                    dim=-1
                )
                fused_frame = torch.matmul(attention_weights, clip)  # Shape: (embed_dim,)
                local_fused_embeddings.append(fused_frame)
        local_fused_embeddings = torch.stack(local_fused_embeddings, dim=0).squeeze(1)  # Shape: (N_frames, embed_dim)
        
        # Compute global context fused embeddings
        global_fused_embeddings = []
        normalized_global = frame_embeddings / (frame_embeddings.norm(dim=-1, keepdim=True) + 1e-6)  # Normalize keys
        for frame in frame_embeddings:
            # Cross-attention between the frame and all frames in the video
            normalized_frame = frame / (frame.norm(dim=-1, keepdim=True) + 1e-6)  # Normalize query
            attention_weights_global = torch.softmax(
                torch.matmul(normalized_frame.unsqueeze(0), normalized_global.transpose(0, 1)) /
                (tau_global * embed_dim ** 0.5),
                dim=-1
            )
            fused_frame_global = torch.matmul(attention_weights_global, frame_embeddings)  # Shape: (embed_dim,)
            global_fused_embeddings.append(fused_frame_global)
        global_fused_embeddings = torch.stack(global_fused_embeddings, dim=0).squeeze(1)  # Shape: (N_frames, embed_dim)


        frame_embeddings = frame_embeddings/frame_embeddings.norm(p=2, dim=-1, keepdim=True)
        local_embeddings = local_fused_embeddings/local_fused_embeddings.norm(p=2, dim=-1, keepdim=True)
        global_embeddings = global_fused_embeddings/global_fused_embeddings.norm(p=2, dim=-1, keepdim=True)
        
        return frame_embeddings, local_embeddings, global_embeddings

        
    def forward(self, pixel_values, input_ids, labels):
        """
        Forward function for computing logits and loss.

        - Labels are converted to clip-level.
        - Logits are computed at the clip level.
        - KL divergence loss is computed at the clip level.

        Returns:
            dict containing loss and logits.
        """
      
        
        pixel_values = pixel_values.squeeze(0)  # Remove extra batch dimension
        frame_embeddings = self.model.get_image_features(pixel_values)[1]
        
        frame_embeddings, local_embeddings, global_embeddings = self.context_fusion_head(frame_embeddings)
        
        logits = []
        
        for q_id in input_ids[0]:
            
            text_embeds = self.model.get_text_features(q_id)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Compute logits for original, local, and global embeddings
            ori_logits_per_text = (
                torch.matmul(text_embeds, frame_embeddings.t().to(text_embeds.device)) * self.model.logit_scale.exp()
                + self.model.logit_bias
            )
            
            local_logits_per_text = (
                torch.matmul(text_embeds, local_embeddings.t().to(text_embeds.device)) * self.model.logit_scale.exp()
                + self.model.logit_bias
            )
            global_logits_per_text = (
                torch.matmul(text_embeds, global_embeddings.t().to(text_embeds.device)) * self.model.logit_scale.exp()
                + self.model.logit_bias
            )
        
            # Combine logits
            new_logits_per_text = 0.9 * ori_logits_per_text + 0.05 * local_logits_per_text + 0.05 * global_logits_per_text
            
            logits.append(new_logits_per_text)

        logits, _ = torch.stack(logits).max(dim=0)  # Aggregate logits

        # Compute clip-wise logits
        logits, clip_variance_loss= self.softmax_pooling_fixed_clips(logits)

        # Convert frame-wise labels to clip-wise labels
        batch_size, num_frames = labels.shape
        clip_size = 8
        num_clips = (num_frames + clip_size - 1) // clip_size  # Number of clips per video
        clip_labels = torch.zeros((batch_size, num_clips), device=labels.device)

        for b in range(batch_size):
            for i in range(num_clips):
                start_idx = i * clip_size
                end_idx = min(start_idx + clip_size, num_frames)
                clip_labels[b, i] = labels[b, start_idx] 

        clip_labels = clip_labels - clip_labels.mean()
        
        
        weighted_logits = logits * clip_labels
        
        loglik = torch.nn.functional.logsigmoid(weighted_logits)
 
        contrastive_loss = -torch.sum(loglik, dim=-1).mean()

        loss = contrastive_loss 

        print("Loss Components - SigLIP Contrastive Loss:", contrastive_loss)

        return {"loss": loss, "logits": logits}
        

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    num_train_epochs=1,
    bf16=True,
    deepspeed="ds_config.json",
    log_level="debug",
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="no",
    evaluation_strategy="no",
    report_to="tensorboard"
)

df = pd.read_csv("siglip_train_data_debiased.csv")
train_dataset = MultiLabelDataset(df=df)

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")

model.gradient_checkpointing_enable()

wrapped_model = CustomModel(model)


trainer = Trainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollator()
)

from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class GradientLoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        # Open the log file in append mode
        self.log_file = open(log_file_path, "a")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Log gradient statistics to the text file
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                self.log_file.write(
                    f"Step {state.global_step} | Parameter: {name} | Gradient Norm: {grad_norm}\n"
                )
        self.log_file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        # Close the log file when training ends
        self.log_file.close()

# Add callback to the trainer
trainer.add_callback(GradientLoggingCallback(log_file_path="./gradient_logs.txt"))

trainer.train()
with deepspeed.zero.GatheredParameters(trainer.model.parameters(), modifier_rank=0):
    trainer.model.save_pretrained(
   "./fine_tuned_siglip_contra_0.9_softmax_pool_debaised",
    safe_serialization=True
    )

    