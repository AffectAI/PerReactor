import os
import torch
import argparse
from tqdm import tqdm
from model import TransformerVAE, PFM
from render import Render
from metric import *
from dataset import get_dataloader
import pandas as pd
import numpy as np
import time

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="../data/react_clean", type=str, help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["train", "val", "test"], required=True)
    parser.add_argument('--general-branch', default="", type=str, help="checkpoint path of the general branch")
    parser.add_argument('--resume-discriminator', default="", type=str, help="checkpoint path of the discriminator of general branch")
    parser.add_argument('--stage', default="person_spcific", type=str, help="select the model to train")
    parser.add_argument('--task', default="person_spcific", type=str, help="task to perform")
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-k', '--k-parameter', default=2, type=int, metavar='N', help='number of unit clips to PFE')
    parser.add_argument('-j', '--num_workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=750, type=int, help="len of video clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--num-samples', default=10, type=int, help="number of sampling from the distribution")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--beta', default=0.1, type=float, help="hyperparameter for the weighting factor beta")
    parser.add_argument('--threads', default=32, type=int, help="num max of threads")
    parser.add_argument('--binarize', action='store_true', help='binarize AUs output from model')
    parser.add_argument('--model', default="baseline", type=str, help="select the model to train: e.g. baseline/cgan")
    parser.add_argument('--loss', default="mse", type=str, help="select the loss for training the discriminator")
    parser.add_argument('--integration', default="summation", type=str, help="select the way to integrate person specific factors into general reaction embeddings")

    args = parser.parse_args()
    return args

# Evaluating
def val(args, model, val_loader, render, binarize=False):
    
    model.eval()

    out_dir = os.path.join(args.outdir, args.split)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    listener_emotion_gt_list = []
    listener_emotion_pred_list = []
    listener_3DMM_pred_list = []
    speaker_emotion_list = []

    for batch_idx, (speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_personal_clip, _, listener_emotion, inappro_listener_emotion, listener_3dmm, inappro_listener_3dmm, listener_references, num_appropriate, num_inappropriate) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            if len(speaker_video_clip.shape) != 1: # if loaded
                speaker_video_clip, speaker_audio_clip = speaker_video_clip[:,:args.clip_length].cuda(), speaker_audio_clip[:,:args.clip_length].cuda()
            speaker_emotion, speaker_3dmm, listener_emotion, listener_3dmm, listener_references = speaker_emotion[:,:args.clip_length].cuda(), speaker_3dmm[:,:args.clip_length].cuda(), listener_emotion[:,:args.clip_length].cuda(), listener_3dmm[:,:args.clip_length].cuda(), listener_references[:,:args.clip_length].cuda()
            num_appropriate = num_appropriate.cuda()
            inappro_listener_emotion, num_inappropriate = inappro_listener_emotion.cuda(), num_inappropriate.cuda()
            inappro_listener_3dmm = inappro_listener_3dmm.cuda()

            listener_personal_clip = listener_personal_clip.cuda()

            num_samples = torch.LongTensor([args.num_samples for i in range(speaker_video_clip.size(0))]).cuda()
        with torch.no_grad():
            if args.model == 'baseline':
                if type(model) == PFM:
                    listener_3dmm_out, listener_emotion_out, _ = model(speaker_video_clip, speaker_audio_clip, listener_personal_clip, num_samples)
                else:
                    listener_3dmm_out, listener_emotion_out, _, _ = model(speaker_video_clip, speaker_audio_clip, num_samples)
            else:                
                # ------------------
                # Evaluate Genorator
                # ------------------
                if type(model) == PFM:
                    # listener_personal_clip = torch.zeros(1,750, 25).cuda()
                    listener_3dmm_out, listener_emotion_out, _ = model(speaker_video_clip, speaker_audio_clip, listener_personal_clip, num_samples)
                else:
                    listener_3dmm_out, listener_emotion_out, _, _ = model(speaker_video_clip, speaker_audio_clip, num_samples)

                
            # binarize first 15 positions
            if binarize:
                listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])
            B = speaker_video_clip.shape[0]
            # if (batch_idx % 100) == 0:
            #     for bs in range(B):
            #         render.rendering_for_fid(out_dir, "{}_b{}_ind{}".format(args.split, str(batch_idx + 1), str(bs + 1)),
            #                 listener_3dmm_out[bs], speaker_video_clip[bs], listener_references[bs], listener_video_clip[bs,:args.clip_length])
            listener_emotion_pred_list.append(listener_emotion_out.view(-1, args.num_samples, listener_emotion_out.shape[1], listener_emotion_out.shape[2]).cpu())
            listener_3DMM_pred_list.append(listener_3dmm_out.view(-1, args.num_samples, listener_3dmm_out.shape[1], listener_3dmm_out.shape[2]).cpu())

            listener_emotion_gt_list.append(listener_emotion.cpu()[0].unsqueeze(dim=0)) # The first listener emotion is the original listener to the given speaker
            speaker_emotion_list.append(speaker_emotion.cpu())

    all_listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim = 0)
    all_listener_3DMM_pred = torch.cat(listener_3DMM_pred_list, dim=0)

    # Save the predicted emotion result of a sample to .csv
    # sample = pd.DataFrame(all_listener_emotion_pred[0][0])
    # sample.to_csv(os.path.join(args.outdir, 'emotion_sample.csv'))
    
    listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim = 0)
    speaker_emotion_gt = torch.cat(speaker_emotion_list, dim = 0)

    # Save the prediction and ground truth.
    np.save(os.path.join(args.outdir, 'all_listener_emotion_pred.npy'), all_listener_emotion_pred.cpu().detach().numpy())
    np.save(os.path.join(args.outdir, 'all_listener_emotion_gt.npy'), listener_emotion_gt.cpu().detach().numpy())
    np.save(os.path.join(args.outdir, 'all_speaker_emotion_gt.npy'), speaker_emotion_gt.cpu().detach().numpy())

    # np.save(os.path.join(args.outdir, 'all_listener_3DMM_pred.npy'), all_listener_3DMM_pred.cpu().detach().numpy())

    if args.task == 'person_specific':
        # neighbour_emotion = 'person_specific_masked_neighbour_emotion_test.npy'
        neighbour_emotion = 'person_specific_masked_neighbour_emotion_train_min.npy'
    elif args.task == 'general':
        # neighbour_emotion = 'neighbour_emotion_test.npy'
        neighbour_emotion = 'neighbour_emotion_train_min.npy'

    print('speaker_emotion_gt shape:', speaker_emotion_gt.shape)
    print('listener_emotion_gt shape:', listener_emotion_gt.shape)
    print('all_listener_emotion_pred shape:', all_listener_emotion_pred.shape)

    print("-----------------Evaluating Metric-----------------")

    p = args.threads
    
    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    TLCC = compute_TLCC_mp(all_listener_emotion_pred, speaker_emotion_gt, p=p)

    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_mp(args, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p, neighbour_emotion=neighbour_emotion)

    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_mp(args, all_listener_emotion_pred, listener_emotion_gt, val_test=args.split, p=p, neighbour_emotion=neighbour_emotion)

    FRDvs = compute_FRDvs(all_listener_emotion_pred)
    FRVar  = compute_FRVar(all_listener_emotion_pred)
    smse  = compute_s_mse(all_listener_emotion_pred)
    
    return FRC, FRD, FRDvs, FRVar, smse, TLCC

def main(args):
    checkpoint_path = args.resume

    log_path = os.path.join(args.outdir, 'loss_log.txt')

    if args.task == 'person_specific':
        val_loader = get_dataloader(args, args.split, args.task, load_audio=True, load_video_s=True, load_clip_p=True, load_video_l=True, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True,  load_3dmm_s=True, load_ref=True)
    elif args.task == 'general':
        val_loader = get_dataloader(args, args.split, args.task, load_audio=True, load_video_s=True, load_clip_p=True, load_video_l=True, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True,  load_3dmm_s=True, load_ref=True)
        
    if args.stage == 'person_specific':
        gen = PFM(img_size = args.img_size, audio_dim = args.audio_dim, output_emotion_dim = args.emotion_dim, output_3dmm_dim = args._3dmm_dim, feature_dim = args.feature_dim, seq_len = args.clip_length, general_branch_path=args.general_branch, integration=args.integration, device = args.device, k = args.k_parameter, beta=args.beta)
    elif args.stage == 'general':
        gen = TransformerVAE(img_size = args.img_size, audio_dim = args.audio_dim,  output_3dmm_dim = args._3dmm_dim, output_emotion_dim = args.emotion_dim, feature_dim = args.feature_dim, seq_len = args.clip_length, device = args.device)
    
    if args.resume != '': #  resume from a checkpoint (generator)
        checkpoint_path = args.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        gen.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        gen = gen.cuda()
        render = Render('cuda')
    else:
        render = Render()

    FRC, FRD, FRDvs, FRVar, smse, TLCC = val(args, gen, val_loader, render, binarize=args.binarize)

    result_message = "Metric: | FRC: {:.5f} | FRD: {:.5f} | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | TLCC: {:.5f}".format(FRC, FRD, smse, FRVar, FRDvs, TLCC)
    latex_message = "Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format(FRC, FRD, smse, FRVar, FRDvs, TLCC)

    print(result_message)
    print(latex_message)

    with open(log_path, 'a') as log_file:
        log_file.write('%s\n' % result_message)
        log_file.write('%s\n' % latex_message)


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '32'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)

