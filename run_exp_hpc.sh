#!/bin/bash
#SBATCH --job-name=dm_350_400
#SBATCH --gres gpu:0
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12

export MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
export SAMPLE_FLAGS="--batch_size 1 --timestep_respacing ddim1000 --use_ddim True"

nvidia-smi

python adam/generate_brats_healthy_volume.py \
    --classifier_scale 100 --noise_level 500 --skip_healthy_slices True \
    --root_dir=/l/users/fadillah.maani/BraTS2023/Adult-Glioma/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --save_dir=/l/users/fadillah.maani/BraTS2023/Adult-Glioma/generated-mris \
    --sample_start_id=350 --sample_end_id=400 \
    --json_filenames=filenames.json $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS