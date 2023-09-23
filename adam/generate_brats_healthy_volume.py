"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
import sys
import json

sys.path.append(".")

from guided_diffusion.bratsloader import BRATSDatasetFullVolume
import torch.nn.functional as F
import numpy as np
import torch as th
from tqdm import tqdm
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()
    
    list_study_names = json.loads(open(args.json_filenames, "r").read())
    end_id = int(args.sample_end_id) if args.sample_end_id else len(list_study_names)
    list_study_names = list_study_names[args.sample_start_id:end_id]
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu",)
    )
    
    model.to("cuda")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    
    classifier.load_state_dict(
        th.load(args.classifier_path, map_location="cpu",)
    )
    
    classifier.to("cuda")
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    
    def cond_fn(x, t,  y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale
    
    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    os.makedirs(args.save_dir, exist_ok = True)
    
    print('-'*30)
    print('sample_start_id', args.sample_start_id)
    print('sample_end_id', args.sample_end_id)
    for study_name in list_study_names:
        mri_path = os.path.join(args.root_dir, study_name)
        
        ds = BRATSDatasetFullVolume(mri_path, skip_healthy_slices=args.skip_healthy_slices)
        datal = th.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=4,
        )
        
        print(study_name)
        print(f'{len(ds)} slices')
        
        save_path = os.path.join(args.save_dir, f'{ds.study_name}.npy')
        if os.path.exists(save_path):
            continue
        
        assert args.batch_size == 1
        
        list_pred = []
        list_orig = []
        list_axial_index = []
        list_vmin = []
        list_vmax = []
        for img in tqdm(datal):
            model_kwargs = {}
            
            if args.class_cond:
                classes = th.randint( # I think this guy is always 0, because low=0 (incluse) and high=1 (exclusive)
                    low=0, high=1, size=(args.batch_size,),
                    device="cuda",
                ) 
                model_kwargs["y"] = classes
            
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            
            sample, x_noisy, org = sample_fn(
                model_fn,
                (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device="cuda",
                noise_level=args.noise_level,
            )
            
            th.cuda.synchronize()
            th.cuda.current_stream().synchronize()
            
            pred = sample.detach().cpu().numpy()
            orig = img[0].detach().cpu().numpy()
            
            list_pred.append(pred)
            list_orig.append(orig)
            list_axial_index.append(img[2].detach().cpu().numpy())
            list_vmin.append(img[3])
            list_vmax.append(img[4])
        
        pred_volume = []
        for seqtype in ['t1n', 't1c', 't2w', 't2f']:
            pred_volume.append(ds.volumes[seqtype].copy())
        pred_volume = np.array(pred_volume).astype(np.float32)
        
        for i in range(len(list_pred)):
            pred = list_pred[i]
            orig = list_orig[i]
            vmin = list_vmin[i]
            vmax = list_vmax[i]
            
            axial_index = list_axial_index[i]
            
            vmin = [x.detach().cpu().numpy() for x in vmin]
            vmin = np.array(vmin).transpose(1,0)[:,:,None,None]
            
            vmax = [x.detach().cpu().numpy() for x in vmax]
            vmax = np.array(vmax).transpose(1,0)
            vmax = np.where(vmax > 0, vmax, 1)[:,:,None,None]
            
            pred = np.clip(pred, 0., 1.)
            pred = pred * (vmax - vmin) + vmin
            
            pred_volume[:, :, :, axial_index[0]] = pred[0, :, 8:-8, 8:-8].copy()
        
        pred_volume = pred_volume.astype(np.int16)
        np.save(save_path, pred_volume)

def create_argparser():
    defaults = dict(
        root_dir="",
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path="weights/brats050000.pt",
        classifier_path="weights/modelbratsclass020000.pt",
        classifier_scale=100,
        noise_level=500,
        skip_healthy_slices=True,
        save_dir=None,
        json_filenames='filenames.json',
        sample_start_id=0,
        sample_end_id=None,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

'''
export MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
export SAMPLE_FLAGS="--batch_size 1 --timestep_respacing ddim1000 --use_ddim True"
'''

# CUDA_VISIBLE_DEVICES=2 python adam/generate_brats_healthy_volume.py --dataset brats --data_dir=/share/sda/adam/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00680-001 --skip_healthy_slices True --classifier_scale 100 --noise_level 500 --save_dir samples $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 

'''
CUDA_VISIBLE_DEVICES=2 python adam/generate_brats_healthy_volume.py --classifier_scale 100 --noise_level 500 --skip_healthy_slices True --root_dir=/share/sda/adam/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData --save_dir=/share/sda/adam/generated-mris --json_filenames=filenames.json --sample_start_id=0 --sample_end_id=50 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
CUDA_VISIBLE_DEVICES=3 python adam/generate_brats_healthy_volume.py --classifier_scale 100 --noise_level 500 --skip_healthy_slices True --root_dir=/share/sda/adam/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData --save_dir=/share/sda/adam/generated-mris --json_filenames=filenames.json --sample_start_id=50 --sample_end_id=100 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
CUDA_VISIBLE_DEVICES=0 python adam/generate_brats_healthy_volume.py --classifier_scale 100 --noise_level 500 --skip_healthy_slices True --root_dir=/share/sda/adam/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData --save_dir=/share/sda/adam/generated-mris --json_filenames=filenames.json --sample_start_id=100 --sample_end_id=150 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
CUDA_VISIBLE_DEVICES=1 python adam/generate_brats_healthy_volume.py --classifier_scale 100 --noise_level 500 --skip_healthy_slices True --root_dir=/share/sda/adam/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData --save_dir=/share/sda/adam/generated-mris --json_filenames=filenames.json --sample_start_id=150 --sample_end_id=200 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
'''