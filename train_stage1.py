import os
import time
import numpy as np
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from model import TransformerVAE, Appropriateness_Discriminator, Frame_Discrminator, Discriminator_IM_Cat, Discriminator_IM_Sum
from utils import AverageMeter, accuracy, binary_accuracy
from render import Render
from dataset import get_dataloader
from model.losses import VAELoss, div_loss_multi, DisLoss, temporal_loss, GenLoss

from torch.utils.tensorboard import SummaryWriter 

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="../data/react_clean", type=str, help="dataset path")
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('--resume-discriminator', default="", type=str, help="checkpoint path of the discriminator of general branch")
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--num_workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--optimizer-eps', default=1e-8, type=float)
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=256, type=int, help="len of video clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--render', action='store_true', help='w/ or w/o render')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--div-p', default=10, type=float, help="hyperparameter for div-loss")
    parser.add_argument('--tem-p', default=0.002, type=float, help="hyperparameter for tem-loss")
    parser.add_argument('--real-p', default=0.00001, type=float, help="hyperparameter for realism loss")
    parser.add_argument('--appr-p', default=1, type=float, help="hyperparameter for appropriateness loss")
    parser.add_argument('--mute-tensorboard', action="store_true", help="whether use tensorboard")
    parser.add_argument('--model', default="baseline", type=str, help="select the model to train: e.g. baseline/cgan")
    parser.add_argument('--loss', default="mse", type=str, help="select the loss for training the discriminator")

    args = parser.parse_args()
    return args

# Train
# def train(args, model, train_loader, optimizer, criterion, epoch, writer=None, D=None, render=None):
def train(args, gen, train_loader, optimizer, criterion, epoch, writer=None, d_f=None, d_a=None, render=None):
    gen_losses = AverageMeter()
    g_f_losses = AverageMeter()
    g_a_losses = AverageMeter()
    d_f_losses = AverageMeter()
    d_a_losses = AverageMeter()
    kld_losses = AverageMeter() # KL divergence
    div_losses = AverageMeter() # Diversity
    con_losses = AverageMeter() # Constraints on intensities of arousal and valence
    d_f_accies = AverageMeter()
    tem_losses = AverageMeter() # Motion smoothness loss
    rec_losses = AverageMeter() # Reconstruction loss for baseline

    if args.model != 'baseline':
        if d_f != None:
            d_f.train()

        if d_a != None:
            d_a.train()

    gen.train()

    for batch_idx, (speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, _, _, _, listener_emotion, inappro_listener_emotion, listener_3dmm, inappro_listener_3dmm, listener_references,  num_appropriate, num_inappropriate) in enumerate(tqdm(train_loader)):
        start_time = time.time()
        
        if torch.cuda.is_available():
            speaker_video_clip, speaker_audio_clip,  listener_emotion, inappro_listener_emotion, listener_3dmm, listener_references, num_appropriate, num_inappropriate = \
                speaker_video_clip.cuda(), speaker_audio_clip.cuda(),  listener_emotion.cuda(), inappro_listener_emotion.cuda(), listener_3dmm.cuda(), listener_references.cuda(), num_appropriate.cuda(), num_inappropriate.cuda()

            speaker_emotion, speaker_3dmm, inappro_listener_3dmm = speaker_emotion.cuda(), speaker_3dmm.cuda(), inappro_listener_3dmm.cuda()

        # For binary classification
        num_samples = num_appropriate - num_inappropriate
        for index in torch.where(num_samples == 0):
            num_samples[index] = 2
        
        if args.model == 'baseline':
            listener_3dmm_out, listener_emotion_out, _, distribution = gen(speaker_video_clip, speaker_audio_clip, num_samples)

            gen_loss, rec_loss, kld_loss = criterion(listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out,
                                                distribution, num_appropriate, num_samples)

            div_loss = div_loss_multi(listener_3dmm_out, num_samples) + div_loss_multi(listener_emotion_out, num_samples)
            t_loss = temporal_loss(listener_3dmm_out)

            gen_loss = gen_loss + args.div_p * div_loss + args.tem_p * t_loss

            gen_losses.update(gen_loss.data.item(), speaker_video_clip.size(0))
            rec_losses.update(rec_loss.data.item(), speaker_video_clip.size(0))
            kld_losses.update(kld_loss.data.item(), speaker_video_clip.size(0))
            div_losses.update(div_loss.data.item(), speaker_video_clip.size(0))

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=gen.parameters(), max_norm=5, norm_type=2)

            g_grad_values = collect_grad_value_(gen.parameters())

            optimizer.step()

            with torch.no_grad():
                if args.render:
                    val_path = os.path.join(args.outdir, 'results_videos', 'train')
                    if not os.path.exists(val_path):
                        os.makedirs(val_path)
                    B = speaker_video_clip.shape[0]
                    if (batch_idx % 3000) == 0:
                        for bs in range(1):
                            render.rendering(val_path, "e{}_b{}_ind{}".format(str(epoch + 1), str(batch_idx + 1), str(bs + 1)),
                                    listener_3dmm_out[bs], speaker_video_clip[bs], listener_references[bs])
            
            # For tensorboard
            const_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            g_f_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            g_a_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_f_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_f_real_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_f_fake_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_f_acc = torch.tensor(0.0).to(listener_emotion.get_device())
            d_a_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_a_real_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_a_fake_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_a_inap_loss = torch.tensor(0.0).to(listener_emotion.get_device())
            d_a_acc = torch.tensor(0.0).to(listener_emotion.get_device())
            d_f_grad_values = None
            d_a_grad_values = None
        else:
            # ---------------------------------------------------------------------------------------------
            # Train Genorator   0 - fake & real but inappropriate reactions, 1 - real appropriate reactions
            # ---------------------------------------------------------------------------------------------
            optimizer[0].zero_grad()

            listener_3dmm_out, listener_emotion_out, _, distribution = gen(speaker_video_clip, speaker_audio_clip, num_samples)

            if d_f != None:
                label_f = torch.cuda.FloatTensor(num_samples.sum()*listener_3dmm_out.shape[1], 1).fill_(1.0)
                valid_f = d_f(listener_3dmm_out.view(-1, listener_3dmm_out.shape[-1]))
            else:
                label_f = None
                valid_f = None

            if d_a != None:
                label_a = torch.cuda.FloatTensor(num_samples.sum(), 1).fill_(1.0)
                valid_a = d_a(speaker_emotion=speaker_emotion, listener_emotion=listener_emotion_out, speaker_3dmm=speaker_3dmm, listener_3dmm=listener_3dmm_out, repeat_interleave=num_samples)
            else:
                label_a = None
                valid_a = None

            gen_loss, g_a_loss, kld_loss, const_loss, g_f_loss = criterion[0](valid_f, label_f, valid_a, label_a, distribution, listener_emotion, listener_emotion_out, num_appropriate, num_samples)

            div_loss = div_loss_multi(listener_3dmm_out, num_samples) + div_loss_multi(listener_emotion_out, num_samples)
            t_loss = temporal_loss(listener_3dmm_out)

            gen_loss = gen_loss + args.div_p * div_loss + args.tem_p * t_loss

            gen_losses.update(gen_loss.data.item(), num_samples.sum())
            kld_losses.update(kld_loss.data.item(), speaker_video_clip.size(0))
            con_losses.update(const_loss.data.item(), num_samples.sum())
            div_losses.update(div_loss.data.item(), num_samples.sum())
            g_f_losses.update(g_f_loss.data.item(), num_samples.sum())
            g_a_losses.update(g_a_loss.data.item(), num_samples.sum())
            tem_losses.update(t_loss.data.item(), num_samples.sum())

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=gen.parameters(), max_norm=5, norm_type=2)

            g_grad_values = collect_grad_value_(gen.parameters())
            optimizer[0].step()

            # -------------------------
            # Train Frame Discriminator
            # -------------------------
            if d_f != None:
                optimizer[1].zero_grad()
                real = torch.cuda.FloatTensor(num_appropriate.sum()*listener_3dmm.shape[1], 1).fill_(1.0)
                fake = torch.cuda.FloatTensor(num_samples.sum()*listener_3dmm_out.shape[1], 1).fill_(0.0)

                pred_real = d_f(listener_3dmm.view(-1, listener_3dmm.shape[-1]))
                d_f_real_loss = criterion[1](pred_real, real)

                pred_fake = d_f(listener_3dmm_out.detach().view(-1, listener_3dmm_out.shape[-1]))
                d_f_fake_loss = criterion[1](pred_fake, fake)

                target = torch.cat([real, fake])
                output = torch.cat([pred_real, pred_fake])

                d_f_acc = binary_accuracy(output, target)
                d_f_loss = criterion[1](output, target)

                d_f_accies.update(d_f_acc.item(), len(target))
                d_f_losses.update(d_f_loss.data.item(), len(target))

                d_f_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=d_f.parameters(), max_norm=5, norm_type=2)
                d_f_grad_values = collect_grad_value_(d_f.parameters())
                optimizer[1].step()
            else:
                d_f_real_loss = torch.tensor(0.0).to(listener_emotion.get_device())
                d_f_fake_loss = torch.tensor(0.0).to(listener_emotion.get_device())
                d_f_acc = 0.
                d_f_loss = torch.tensor(0.0).to(listener_emotion.get_device())
                d_f_grad_values = None

            # -----------------------------------
            # Train Appropriateness Discriminator
            # -----------------------------------
            if d_a != None:
                optimizer[2].zero_grad()
                real = torch.cuda.FloatTensor(num_appropriate.sum(), 1).fill_(1.0)
                fake = torch.cuda.FloatTensor(num_samples.sum(), 1).fill_(0.0)
                inap = torch.cuda.FloatTensor(num_inappropriate.sum(), 1).fill_(0.0)

                pred_real = d_a(speaker_emotion=speaker_emotion, listener_emotion=listener_emotion, speaker_3dmm=speaker_3dmm, listener_3dmm=listener_3dmm, repeat_interleave=num_appropriate)
                d_a_real_loss = criterion[2](pred_real, real)

                pred_fake = d_a(speaker_emotion=speaker_emotion, listener_emotion=listener_emotion_out.detach(), speaker_3dmm=speaker_3dmm, listener_3dmm=listener_3dmm_out.detach(), repeat_interleave=num_samples)
                d_a_fake_loss = criterion[2](pred_fake, fake)

                pred_inap = d_a(speaker_emotion=speaker_emotion, listener_emotion=inappro_listener_emotion, speaker_3dmm=speaker_3dmm, listener_3dmm=inappro_listener_3dmm, repeat_interleave=num_inappropriate)
                d_a_inap_loss = criterion[2](pred_inap, inap)

                target = torch.cat([real, fake, inap])
                output = torch.cat([pred_real, pred_fake, pred_inap])

                d_a_acc = binary_accuracy(output, target)
                d_a_loss = criterion[2](output, target)

                d_a_losses.update(d_a_loss.data.item(), len(target))

                d_a_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=d_a.parameters(), max_norm=5, norm_type=2)
                d_a_grad_values = collect_grad_value_(d_a.parameters())

                optimizer[2].step()
            else:
                d_a_real_loss = torch.tensor(0.0).to(listener_emotion.get_device())
                d_a_fake_loss = torch.tensor(0.0).to(listener_emotion.get_device())
                d_a_inap_loss = torch.tensor(0.0).to(listener_emotion.get_device())
                d_a_acc = 0.
                d_a_loss = torch.tensor(0.0).to(listener_emotion.get_device())
                d_a_grad_values = None

            with torch.no_grad():
                if args.render:
                    val_path = os.path.join(args.outdir, 'results_videos', 'train')
                    if not os.path.exists(val_path):
                        os.makedirs(val_path)
                    B = speaker_video_clip.shape[0]
                    if (batch_idx % 3000) == 0:
                        for bs in range(B):
                            render.rendering(val_path, "e{}_b{}_ind{}".format(str(epoch + 1), str(batch_idx + 1), str(bs + 1)),
                                    listener_3dmm_out[bs], speaker_video_clip[bs], listener_references[bs])

        end_time = time.time()

        time_taken = end_time - start_time
        # print(f"Time taken to execute the loop: {time_taken} seconds")

        if writer is not None:
            iters = epoch * len(train_loader) + batch_idx
            tensorboard_log(writer, iters, gen_loss, kld_loss, const_loss,
                            div_loss, g_f_loss, g_a_loss,
                            d_f_loss, d_f_real_loss, d_f_fake_loss, d_f_acc,
                            d_a_loss, d_a_real_loss, d_a_fake_loss, d_a_inap_loss, d_a_acc,
                            g_grad_values, d_f_grad_values, d_a_grad_values)

    if args.model == 'baseline':
        return gen_losses.avg, rec_losses.avg, kld_losses.avg, div_losses.avg
    else:
        return gen_losses.avg, d_f_losses.avg, d_a_losses.avg, div_losses.avg, tem_losses.avg

def collect_grad_value_(parameters):
    grad_values = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        grad_values.append(p.grad.data.abs().mean().item())
    grad_values = np.array(grad_values)
    return grad_values

def tensorboard_log(writer, iter, gen_loss, kld_loss, const_loss,
                    div_loss, g_f_loss, g_a_loss,
                    d_f_loss, d_f_real_loss, d_f_fake_loss, d_f_acc,
                    d_a_loss, d_a_real_loss, d_a_fake_loss, d_a_inap_loss, d_a_acc,
                    g_grad_values, d_f_grad_values, d_a_grad_values):

    writer.add_scalar('gen_loss', gen_loss, iter)
    writer.add_scalar('kld_loss', kld_loss, iter)
    writer.add_scalar('const_loss', const_loss, iter)
    writer.add_scalar('div_loss', div_loss, iter)
    writer.add_scalar('g_f_loss', g_f_loss, iter)
    writer.add_scalar('g_a_loss', g_a_loss, iter)
    writer.add_scalar('d_f_loss', d_f_loss, iter)
    writer.add_scalar('d_f_real_loss', d_f_real_loss, iter)
    writer.add_scalar('d_f_fake_loss', d_f_fake_loss, iter)
    writer.add_scalar('d_f_acc', d_f_acc, iter)
    writer.add_scalar('d_a_loss', d_a_loss, iter)
    writer.add_scalar('d_a_real_loss', d_a_real_loss, iter)
    writer.add_scalar('d_a_fake_loss', d_a_fake_loss, iter)
    writer.add_scalar('d_a_inap_loss', d_a_inap_loss, iter)
    writer.add_scalar('d_a_acc', d_a_acc, iter)

    if g_grad_values is not None:
        writer.add_scalar('g_grad_mean', g_grad_values.mean(), iter)
        writer.add_scalar('g_grad_max', g_grad_values.max(), iter)
    
    if d_f_grad_values is not None:
        writer.add_scalar('d_f_grad_mean', d_f_grad_values.mean(), iter)
        writer.add_scalar('d_f_grad_max', d_f_grad_values.max(), iter)

    if d_a_grad_values is not None:
        writer.add_scalar('d_a_grad_mean', d_a_grad_values.mean(), iter)
        writer.add_scalar('d_a_grad_max', d_a_grad_values.max(), iter)

# Validation
def val(args, gen, val_loader, render, epoch):
    gen.eval()

    for batch_idx, (speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, _, _, _, listener_emotion, inappro_listener_emotion, listener_3dmm, inappro_listener_3dmm, listener_references, num_appropriate, num_inappropriate) in enumerate(tqdm(val_loader)):
        
        if torch.cuda.is_available():
            speaker_video_clip, speaker_audio_clip, listener_emotion, inappro_listener_emotion, listener_3dmm, listener_references, num_appropriate, num_inappropriate = \
                speaker_video_clip.cuda(), speaker_audio_clip.cuda(), listener_emotion.cuda(), inappro_listener_emotion.cuda(), listener_3dmm.cuda(), listener_references.cuda(), num_appropriate.cuda(), num_inappropriate.cuda()

            speaker_emotion, speaker_3dmm, inappro_listener_3dmm = speaker_emotion.cuda(), speaker_3dmm.cuda(), inappro_listener_3dmm.cuda()

            # For binary classification
            num_samples = num_appropriate - num_inappropriate
            for index in torch.where(num_samples == 0):
                num_samples[index] = 2

        with torch.no_grad():
            if (batch_idx % 500) != 0:
                continue

            if args.model == 'baseline':
                listener_3dmm_out, _, _, _ = gen(speaker_video_clip, speaker_audio_clip, num_samples)
            else:
                # args.model = 'cgan'
                listener_3dmm_out, _, _, _ = gen(speaker_video_clip, speaker_audio_clip, num_samples)    

            if args.render:
                val_path = os.path.join(args.outdir, 'results_videos', 'val')
                if not os.path.exists(val_path):
                    os.makedirs(val_path)
                B = speaker_video_clip.shape[0]
                if (batch_idx % 500) == 0:
                    for bs in range(B):
                        render.rendering(val_path, "e{}_b{}_ind{}".format(str(epoch + 1), str(batch_idx + 1), str(bs + 1)),
                                listener_3dmm_out[bs], speaker_video_clip[bs], listener_references[bs])

def save_checkpoint(args, epoch, gen, d_f=None, d_a=None, optimizer=None):
    if args.model == 'baseline':
        checkpoint = {
            'state_dict': gen.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.outdir, f'gen_epoch{epoch+1}_checkpoint.pth'))

    else:
        # Save the generator
        checkpoint = {
            'state_dict': gen.state_dict(),
            'optimizer': optimizer[0].state_dict(),
        }
        if not os.path.exists(os.path.join(args.outdir, 'generator')):
            os.makedirs(os.path.join(args.outdir, 'generator'))
        torch.save(checkpoint, os.path.join(args.outdir, 'generator', f'gen_epoch{epoch+1}_checkpoint.pth'))
        
        # Save the d_f
        if d_f != None:
            checkpoint = {
                'state_dict': d_f.state_dict(),
                'optimizer': optimizer[1].state_dict(),
            }
            if not os.path.exists(os.path.join(args.outdir, 'discriminator')):
                os.makedirs(os.path.join(args.outdir, 'discriminator'))
            torch.save(checkpoint, os.path.join(args.outdir, 'discriminator', f'd_f_epoch{epoch+1}_checkpoint.pth'))

        # Save the d_a
        if d_a != None:
            checkpoint = {
                'state_dict': d_a.state_dict(),
                'optimizer': optimizer[2].state_dict(),
            }
            if not os.path.exists(os.path.join(args.outdir, 'discriminator')):
                os.makedirs(os.path.join(args.outdir, 'discriminator'))
            torch.save(checkpoint, os.path.join(args.outdir, 'discriminator', f'd_a_epoch{epoch+1}_checkpoint.pth'))
    
def main(args):
    start_epoch = 0
    
    if args.mute_tensorboard:
        writer = None
    else:
        writer = SummaryWriter(args.outdir)

    train_loader = get_dataloader(args, "train", "general", load_audio=True, load_video_s=True, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True, load_3dmm_s=True, load_ref=True)
    val_loader = get_dataloader(args, "val", "general", load_audio=True, load_video_s=True, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True, load_3dmm_s=True, load_ref=True)
    
    gen = TransformerVAE(img_size = args.img_size, audio_dim = args.audio_dim,  output_3dmm_dim = args._3dmm_dim, output_emotion_dim = args.emotion_dim, feature_dim = args.feature_dim, seq_len = args.clip_length, device = args.device)
    
    if args.model == 'baseline':
        d_f = None
        d_a = None
        criterion = VAELoss(args.kl_p)
        optimizer = optim.AdamW(gen.parameters(), betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        d_f = Frame_Discrminator()
        d_a = Appropriateness_Discriminator()
        # d_a = None
        criterion = [GenLoss(args.kl_p, args.real_p, args.appr_p), DisLoss('bce'), DisLoss('mse')]

        optimizer = []
        optimizer.append(optim.AdamW(gen.parameters(), betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=args.weight_decay))

        if d_f != None:
            optimizer.append(optim.AdamW(d_f.parameters(), betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=args.weight_decay))
        else:
            optimizer.append(None)

        if d_a != None:
            optimizer.append(optim.AdamW(d_a.parameters(), betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=args.weight_decay))
        else:
            optimizer.append(None)

    if torch.cuda.is_available():
        gen = gen.cuda()
        d_f = d_f.cuda() if d_f != None else None
        d_a = d_a.cuda() if d_a != None else None
        render = Render('cuda')
    else:
        render = Render()

    log_path = os.path.join(args.outdir, 'loss_log.txt')
    with open(log_path, 'a') as log_file:
        now = time.strftime('%c')
        log_file.write('================ Training Loss (%s) ================\n' % now)
    
    for epoch in range(start_epoch, args.epochs):
        if args.model == 'baseline':
            gen_loss, rec_loss, kld_loss, div_loss = train(args, gen, train_loader, optimizer, criterion, epoch, writer, None, None, render=render)
            train_message = "Epoch: {} gen_loss: {:.5f} rec_loss: {:.5f} kld_loss: {:.5f} div_loss: {:.5f}".format(epoch+1, gen_loss, rec_loss, kld_loss, div_loss)
            print(train_message)
        elif args.model == 'cgan':
            gen_loss, d_f_loss, d_a_loss, div_loss, tem_loss = train(args, gen, train_loader, optimizer, criterion, epoch, writer, d_f, d_a, render)
            train_message = "Epoch: {} gen_loss: {:.5f} d_f_loss: {:.5f} d_a_loss: {:.5f} div_loss: {:.5f}".format(epoch+1, gen_loss, d_f_loss, d_a_loss, div_loss, tem_loss)
            print(train_message)

        if (epoch+1) % 1 == 0:
            val(args, gen, val_loader, render, epoch)

        with open(log_path, 'a') as log_file:
            log_file.write('%s\n' % train_message)

        save_checkpoint(args, epoch, gen, d_f, d_a, optimizer)

# ---------------------------------------------------------------------------------

if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '32'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)