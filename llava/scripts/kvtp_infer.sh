#!/usr/bin/bash

# video tasks
accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_vid_kvtp \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,prune_method=prumerge,predictor_path=kvtp_predictor,conv_template=qwen_1_5,max_frames_num=128,mm_spatial_pool_mode=average \
--tasks videomme,nextqa_mc_test,egoschema \
--batch_size 1 \
--seed 42 \
--log_samples \
--log_samples_suffix llava_vid_kvtp \
--output_path ./test_logs 