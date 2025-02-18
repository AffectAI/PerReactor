import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ====================== Loss for the baseline model ======================

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"

def get_compared_index(gt_emotion, gt_3dmm, pred_emotion, pred_3dmm, num_appropriate, batch_size=4, num_samples=None):
    compared_index = []
    start_index = 0

    mse = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for i in range(batch_size):
            # Find the closest ground truth for each sample
            for j in range(num_samples[i]):
                pred_emotion_sample = pred_emotion[num_samples[:i].sum()+j]
                pred_3dmm_sample_1 = pred_3dmm[num_samples[:i].sum()+j, :, :52]
                pred_3dmm_sample_2 = pred_3dmm[num_samples[:i].sum()+j, :, 52:]

                # [256, 25] ---> [num_appropriate[i], 256, 25]               
                pred_emotion_sample = pred_emotion_sample.expand(num_appropriate[i], pred_emotion_sample.shape[0], pred_emotion_sample.shape[1])
                pred_3dmm_sample_1 = pred_3dmm_sample_1.expand(num_appropriate[i], pred_3dmm_sample_1.shape[0], pred_3dmm_sample_1.shape[1])
                pred_3dmm_sample_2 = pred_3dmm_sample_2.expand(num_appropriate[i], pred_3dmm_sample_2.shape[0], pred_3dmm_sample_2.shape[1])
                
                # rec_loss = torch.mean(mse(pred_emotion_sample, gt_emotion[start_index:start_index+num_appropriate[i],:,:]), dim=(1, 2))
                rec_loss = torch.mean(mse(pred_emotion_sample, gt_emotion[start_index:start_index+num_appropriate[i],:,:]), dim=(1, 2)) + \
                            torch.mean(mse(pred_3dmm_sample_1, gt_3dmm[start_index:start_index+num_appropriate[i],:,:52]), dim=(1, 2)) + \
                            10*torch.mean(mse(pred_3dmm_sample_2, gt_3dmm[start_index:start_index+num_appropriate[i],:,52:]), dim=(1, 2))

                # rec_loss = torch.mean(mse(pred_emotion_sample, gt_emotion[start_index:start_index+num_appropriate[i],:,15:17]), dim=(1, 2))

                # Get the ground truth with the smallest loss for the sample
                loc_index = torch.argmin(rec_loss)
                compared_index.append(start_index+loc_index)

            start_index += num_appropriate[i]
            
    return compared_index


class VAELoss(nn.Module):
    def __init__(self, kl_p=0.0002):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, gt_emotion, gt_3dmm, pred_emotion, pred_3dmm, distribution, num_appropriate, num_samples):
        b = num_appropriate.shape[0]
        compared_index = get_compared_index(gt_emotion, gt_3dmm, pred_emotion, pred_3dmm, num_appropriate, b, num_samples)
        
        # rec_loss = self.mse(pred_emotion, gt_emotion[compared_index,:,:])
        rec_loss = self.mse(pred_emotion, gt_emotion[compared_index,:,:]) + \
                        self.mse(pred_3dmm[:,:, :52], gt_3dmm[compared_index,:, :52]) + \
                        10*self.mse(pred_3dmm[:,:, 52:], gt_3dmm[compared_index,:, 52:])
        
        kld_loss = torch.tensor(0.0).to('cuda')
        if distribution != None:        
            mu_ref = torch.zeros_like(distribution[0].loc).to('cuda')
            scale_ref = torch.ones_like(distribution[0].scale).to('cuda')
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

            kld_loss = 0
            for t in range(len(distribution)):
                kld_loss += self.kl_loss(distribution[t], distribution_ref)
            kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss

        return loss, rec_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"

# ======================= Diversity Loss =======================

def div_loss(Y_1, Y_2):
    # Y_1: (B, T, C)
    # Y_2: (B, T, C)
    loss = torch.tensor(0.0).to(Y_1.get_device())
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss

def div_loss_multi_f(Y, num_samples=3):
    # num_samples is a fixed number
    # Y: (B*num_samples, T, C)
    index = []
    loss = torch.tensor(0.0).to(Y.get_device())
    b,t,c = Y.shape
    b //= num_samples

    for i in range(num_samples):
        index.append([])

    for i in range(b):
        for j in range(num_samples):
            index[j].append(j+i*num_samples)

    for i in range(b):
        for j in range(i+1, num_samples):
            loss += div_loss(Y[index[i]], Y[index[j]])  

    loss /= num_samples*(num_samples-1)/2
    return loss

def div_loss_multi(Y, num_samples=None):
    # num_samples is a 1D tensor
    B = num_samples.shape[0]
    loss = torch.tensor(0.0).to(Y.get_device())
    start_index = 0
    skip_sample = 0

    for bs in range(B):
        n_s = num_samples[bs].item()

        if n_s == 1:
            skip_sample += 1
            start_index += n_s
            continue

        Y_g = Y[start_index: start_index+n_s,:]
        loss += div_loss_multi_f(Y_g, n_s)
        
        start_index += n_s

    if B - skip_sample != 0:
        loss /= (B - skip_sample)

    return loss

# ====================== Temopral Constraints ======================
def temporal_loss(Y):
    t_loss = torch.tensor(0.0).to(Y.get_device())
    for i in range(Y.shape[0]):
        loss = torch.mean(torch.norm(Y[i, 1:, :] - Y[i, :-1, :], dim=1, p=2)**2)
        # num_frames = Y.shape[1]
        # t_loss += loss / num_frames
        t_loss += loss
    t_loss /= Y.shape[0]
    return t_loss

# ====================== Loss for the cgan model ======================

def get_compared_index_emotion(gt_emotion, pred_emotion, num_appropriate, num_samples=None):
    compared_index = []
    start_index = 0

    mse = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for i in range(len(num_samples)):
            # Find the closest ground truth for each sample
            for j in range(num_samples[i]):
                pred_emotion_sample = pred_emotion[num_samples[:i].sum()+j]

                # [256, 25] ---> [num_appropriate[i], 256, 25]
                pred_emotion_sample = pred_emotion_sample.expand(num_appropriate[i], pred_emotion_sample.shape[0], pred_emotion_sample.shape[1])    

                rec_loss = torch.mean(mse(pred_emotion_sample[:,:,15:17], gt_emotion[start_index:start_index+num_appropriate[i],:,15:17]), dim=(1, 2))

                # Get the ground truth with the smallest loss for the sample
                loc_index = torch.argmin(rec_loss)
                compared_index.append(start_index+loc_index)

            start_index += num_appropriate[i]

    return compared_index

class GenLoss_Const(nn.Module):
    def __init__(self, kl_p=0.0002, loss='mse'):
        super(GenLoss_Const, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        if loss == 'mse':
            self.loss = nn.MSELoss(reduction='mean')
        elif loss == 'bce':
            self.loss = nn.BCELoss(reduction='mean')
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, pred, target, distribution, gt_emotion, pred_emotion, num_appropriate, num_samples):
        batch_size = num_appropriate.shape[0]

        # Constraint loss on arousal and valence intensities
        compared_index = get_compared_index_emotion(gt_emotion, pred_emotion, num_appropriate, num_samples)
        const_loss = self.mse(pred_emotion[:,:,15:17], gt_emotion[compared_index,:,15:17]) 

        # Classification loss
        rec_loss = self.loss(pred, target)

        # KL divergence loss
        kld_loss = torch.tensor(0.0).to(pred.get_device())
        if distribution != None:
            mu_ref = torch.zeros_like(distribution[0].loc).to(target.get_device())
            scale_ref = torch.ones_like(distribution[0].scale).to(target.get_device())
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

            for t in range(len(distribution)):
                kld_loss += self.kl_loss(distribution[t], distribution_ref)
            kld_loss = kld_loss / len(distribution)

        # total loss for the generator
        loss = rec_loss + self.kl_p * kld_loss + const_loss

        return loss, rec_loss, kld_loss, const_loss

    def __repr__(self):
        return "GenLoss_Const()"

# Consist of g_a_loss, kld_loss, const_loss, g_f_loss
class GenLoss(nn.Module):
    def __init__(self, kl_p=0.0002, real_p=0.00001, loss='mse'):
        super(GenLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')
        self.kl_loss = KLLoss()
        self.kl_p = kl_p
        self.real_p = real_p

    def forward(self, valid_f, label_f, valid_a, label_a, distribution, gt_emotion, pred_emotion, num_appropriate, num_samples):
        # Constraint loss on arousal and valence intensities
        compared_index = get_compared_index_emotion(gt_emotion, pred_emotion, num_appropriate, num_samples)
        const_loss = self.mse(pred_emotion[:,:,15:17], gt_emotion[compared_index,:,15:17]) 

        # frame-level realism loss
        if valid_f != None:
            g_f_loss = self.bce(valid_f, label_f)
        else:
            g_f_loss = torch.tensor(0.0).to('cuda')

        # Classification loss based on d_a
        if valid_a != None:
            g_a_loss = self.mse(valid_a, label_a)
        else:
            g_a_loss = torch.tensor(0.0).to('cuda')

        # KL divergence loss
        kld_loss = torch.tensor(0.0).to('cuda')
        if distribution != None:
            mu_ref = torch.zeros_like(distribution[0].loc).to('cuda')
            scale_ref = torch.ones_like(distribution[0].scale).to('cuda')
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

            for t in range(len(distribution)):
                kld_loss += self.kl_loss(distribution[t], distribution_ref)
            kld_loss = kld_loss / len(distribution)

        # total loss for the generator
        gen_loss = g_a_loss + self.kl_p * kld_loss + const_loss + self.real_p * g_f_loss

        return gen_loss, g_a_loss, kld_loss, const_loss, g_f_loss

    def __repr__(self):
        return "GenLoss()"
    
class DisLoss(nn.Module):
    def __init__(self, loss='mse'):
        super(DisLoss, self).__init__()
        if loss == 'mse':
            self.loss = nn.MSELoss(reduction='mean') 
        elif loss == 'bce':
            self.loss = nn.BCELoss(reduction='mean')
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, pred, target):
        dis_loss = self.loss(pred, target)
    
        return dis_loss