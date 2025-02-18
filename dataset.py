import os
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import random
import pandas as pd
from PIL import Image
import soundfile as sf
import cv2
from torch.utils.data import DataLoader
import torchaudio

torchaudio.set_audio_backend("sox_io")

import av
import math

from decord import VideoReader
from decord import cpu

class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size
        
    def __call__(self, img):

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)
        return img
    
def extract_audio_features(audio_path, fps, n_frames):
    audio, sr = sf.read(audio_path)

    if audio.ndim == 2:
        audio = audio.mean(-1)

    frame_n_samples = int(sr / fps)
    curr_length = len(audio)

    target_length = frame_n_samples * n_frames

    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])

    shifted_n_samples = 0
    curr_feats = []

    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)

    return curr_feats

def custom_collate(batch):
    speaker_video_clip = [item[0] for item in batch]
    speaker_audio_clip = [item[1] for item in batch]
    speaker_emotion = [item[2] for item in batch]
    speaker_3dmm = [item[3] for item in batch]
    listener_video_clip = [item[4] for item in batch]
    listener_personal_clip = [item[5] for item in batch]
    listener_audio_clip = [item[6] for item in batch]
    listener_emotion = [item[7] for item in batch]
    inappro_listener_emotion = [item[8] for item in batch if len(item[8]) != 0]
    listener_3dmm = [item[9] for item in batch]
    inappro_listener_3dmm = [item[10] for item in batch if len(item[10]) != 0]
    listener_references = [item[11] for item in batch]
    num_appropriate = [item[12] for item in batch]
    num_inappropriate = [item[13] for item in batch]
     
    if type(speaker_video_clip[0]) == torch.Tensor:
        speaker_video_clip = torch.stack(speaker_video_clip)
        
    if type(speaker_audio_clip[0]) == torch.Tensor:
        speaker_audio_clip = torch.stack(speaker_audio_clip)
        
    if type(speaker_emotion[0]) == torch.Tensor:
        speaker_emotion = torch.stack(speaker_emotion)
        
    if type(speaker_3dmm[0]) == torch.Tensor:
        speaker_3dmm = torch.stack(speaker_3dmm)
        
    if type(listener_video_clip[0]) == torch.Tensor:
        listener_video_clip = torch.stack(listener_video_clip)
    
    if type(listener_personal_clip[0]) == torch.Tensor:
        listener_personal_clip = torch.stack(listener_personal_clip)
        
    if type(listener_audio_clip[0]) == torch.Tensor:
        listener_audio_clip = torch.cat(listener_audio_clip, dim=0)
        
    if type(listener_emotion[0]) == torch.Tensor:
        listener_emotion = torch.cat(listener_emotion, dim=0)

    if len(inappro_listener_emotion) != 0:
        inappro_listener_emotion = torch.cat(inappro_listener_emotion, dim=0)
    else:
        inappro_listener_emotion = torch.Tensor(inappro_listener_emotion)
        
    if type(listener_3dmm[0]) == torch.Tensor:
        listener_3dmm = torch.cat(listener_3dmm, dim=0)

    if len(inappro_listener_3dmm) != 0:
        inappro_listener_3dmm = torch.cat(inappro_listener_3dmm, dim=0)
    else:
        inappro_listener_3dmm = torch.Tensor(inappro_listener_3dmm)
        
    if type(listener_references[0]) == torch.Tensor:
        listener_references = torch.stack(listener_references, dim=0)

    num_appropriate = torch.LongTensor(num_appropriate)
    num_inappropriate = torch.LongTensor(num_inappropriate)
        
    return speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_personal_clip, listener_audio_clip, listener_emotion, inappro_listener_emotion, listener_3dmm, inappro_listener_3dmm, listener_references, num_appropriate, num_inappropriate


class ReactionDataset(data.Dataset):
    def __init__(self, root_path, split, stage='general', img_size=256, crop_size=224, clip_length=751, personal_clip_length=240, fps=25,
                 load_audio=True, load_video_s=True, load_video_l=True, load_clip_p=True, load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True,
                 repeat_mirrored=True):
        
        self._root_path = root_path     # ../data/react_clean
        self._clip_length = clip_length
        self._personal_clip_length = personal_clip_length
        self._fps = fps
        self._split = split
        self._stage = stage               # either 'general' or 'person_specific'

        self._data_path = os.path.join(self._root_path, self._split)
        
        if self._stage == 'general':
            self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'), header=None, delimiter=',')
            neighbour_emotion_path = os.path.join(root_path, 'neighbour_emotion_'+split+'.npy')
            # self._list_path = pd.read_csv(os.path.join(self._root_path, self._split + '_min.csv'), header=None, delimiter=',')
            # neighbour_emotion_path = os.path.join(root_path, 'neighbour_emotion_'+split+'_min.npy')
        else:
            self._list_path = pd.read_csv(os.path.join(self._root_path, 'person_specific_' + self._split + '.csv'), header=None, delimiter=',')
            # # neighbour_emotion_path = os.path.join(root_path, 'person_specific_neighbour_emotion_'+split+'.npy')
            if self._split == 'train':
                neighbour_emotion_path = os.path.join(root_path, 'person_specific_neighbour_emotion_'+split+'.npy')
            else:
                neighbour_emotion_path = os.path.join(root_path, 'person_specific_masked_neighbour_emotion_'+split+'.npy')
            # self._list_path = pd.read_csv(os.path.join(self._root_path, 'person_specific_' + self._split + '_min.csv'), header=None, delimiter=',')
            # neighbour_emotion_path = os.path.join(root_path, 'person_specific_masked_neighbour_emotion_'+split+'_min.npy')
            
        self._list_path = self._list_path.drop(0)
        self.neighbour_emotion = np.load(neighbour_emotion_path)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_clip_p = load_clip_p
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

        self.dataset_path=os.path.join(root_path, self._split)

        self._audio_path = os.path.join(self.dataset_path, 'Audio_files')
        self._video_path = os.path.join(self.dataset_path, 'Video_files')
        self._emotion_path = os.path.join(self.dataset_path, 'Emotion')
        self._3dmm_path = os.path.join(self.dataset_path, '3D_FV_files')

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1)
        
        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        speaker_path = [path for path in list(self._list_path.values[:, 1])]
        listener_path = [path for path in list(self._list_path.values[:, 2])]

        if self._split in ["val", "test"] or repeat_mirrored:  # training is always mirrored as data augmentation
            speaker_path_tmp = speaker_path + listener_path
            listener_path_tmp = listener_path + speaker_path
            speaker_path = speaker_path_tmp
            listener_path = listener_path_tmp

        self.speaker_path = speaker_path.copy()
        self.listener_path = listener_path.copy()

        self.data_list = [path for path in list(self._list_path.values[:, 1])] + [path for path in list(self._list_path.values[:, 2])]  # the data_list is actually the same as speaker_path

        self._len = len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        # =================== Find Similar Speakers & Appropriate Reactions ===================
        speaker_line = self.neighbour_emotion[index]
        sim_speakers_index = np.where(speaker_line == True)[0]

        if self._stage == 'general':
            all_listeners_index = np.arange(self._len)
            appro_listeners_index = sim_speakers_index
            selected_listener_index = index

            # Make sure the first index is the original one to the speaker
            if len(appro_listeners_index) >= 2:
                new_list = []
                new_list.append(index)
                for e in sim_speakers_index:
                    if e != index:
                        new_list.append(e)
                appro_listeners_index = np.asarray(new_list)

        elif self._stage == 'person_specific':
            if self._split in ['val', 'test']:
                # The original paired listener to the speaker
                selected_listener_index = index
                selected_listener_path = self.listener_path[selected_listener_index]
                site, group, pid, _ = selected_listener_path.split('/')

                # Get the indexes of all clips belonging to that listener
                all_listeners_index = np.asarray([i for i, path in enumerate(self.listener_path) if '/'.join([site, group, pid]) in path])
                appro_listeners_index = np.intersect1d(all_listeners_index, sim_speakers_index) # All the index of appropriate reactions belonging to that listener

                # Make sure the first index is the original one to the speaker
                if len(appro_listeners_index) >= 2:
                    new_list = []
                    new_list.append(selected_listener_index)
                    for e in appro_listeners_index:
                        if e != index:
                            new_list.append(e)
                    appro_listeners_index = np.asarray(new_list)
            else:
                # For training, we randomly select a listener and all appropriate listener reactions belonging to this listener
                selected_listener_index = np.random.choice(sim_speakers_index)
                selected_listener_path = self.listener_path[selected_listener_index]
                site, group, pid, _ = selected_listener_path.split('/')

                # Get the indexes of all clips belonging to that listener
                all_listeners_index = np.asarray([i for i, path in enumerate(self.listener_path) if '/'.join([site, group, pid]) in path])
                appro_listeners_index = np.intersect1d(all_listeners_index, sim_speakers_index)

        inappro_listeners_index = np.setdiff1d(all_listeners_index, appro_listeners_index)

        # For binary classification performed by the discriminator
        # The ratio among real appropriate: real inappropriate: fake = 2 : 1 : 1
        if len(inappro_listeners_index) > math.ceil(len(appro_listeners_index)/2):
            inappro_listeners_index = np.random.choice(inappro_listeners_index, math.ceil(len(appro_listeners_index)/2), replace=False)

        # For three-class classification performed by the discrimiantor
        # The ratio among real appropriate: real inappropriate: fake = 1 : 1 : 1
        # if len(inappro_listeners_index) > len(appro_listeners_index):
        #     inappro_listeners_index = np.random.choice(inappro_listeners_index, len(appro_listeners_index), replace=False)
        # elif len(inappro_listeners_index) < len(appro_listeners_index) and len(inappro_listeners_index) != 0:
        #     repeat_index = np.random.choice(inappro_listeners_index, len(appro_listeners_index) - len(inappro_listeners_index), replace=True)
        #     inappro_listeners_index = np.concatenate((inappro_listeners_index, repeat_index), axis=0)
            
        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = os.path.join(self._video_path, self.speaker_path[index]+'.mp4')
        listener_video_path = os.path.join(self._video_path, self.listener_path[index]+'.mp4')

        total_length = 751

        cp = random.randint(0, total_length - 1 - self._clip_length) if self._clip_length < total_length else 0

        speaker_video_clip = 0
        if self.load_video_s:
            clip = []

            with open(speaker_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length):
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))

            # shape: [_clip_length, 3, 224, 224]
            speaker_video_clip = torch.cat(clip, dim=0)

        # listener video clip only needed for val/test
        listener_video_clip = 0
        if self.load_video_l:
            clip = []

            with open(listener_video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            for i in range(cp,cp + self._clip_length):
                frame = vr[i]
                img=Image.fromarray(frame.asnumpy())
                img = self._transform(img)
                clip.append(img.unsqueeze(0))

            # shape: [_clip_length, 3, 224, 224]
            listener_video_clip = torch.cat(clip, dim=0)

        # ============================= Load Listener personal clip ==============================
        listener_personal_clip = 0
        # Load the video clip for personality learning
        # if self.load_clip_p:
        #     selected_listener_path = self.listener_path[selected_listener_index]
        #     site, group, pid, _ = selected_listener_path.split('/')
        #     listener_video_path = os.path.join(self._video_path, '/'.join([site, group, pid, '1'])+'.mp4')

        #     clip = []

        #     with open(listener_video_path, 'rb') as f:
        #         vr = VideoReader(f, ctx=cpu(0))
        #     for i in range(0, self._personal_clip_length):  # Only use the first few seconds of the clip
        #         frame = vr[i]
        #         img=Image.fromarray(frame.asnumpy())
        #         img = self._transform(img)
        #         clip.append(img.unsqueeze(0))

        #     # shape: [_personal_clip_length, 3, 224, 224]
        #     listener_personal_clip = torch.cat(clip, dim=0)

        # Or Load the listener emotion clip for personality learning
        if self.load_clip_p:
            selected_listener_path = self.listener_path[selected_listener_index]
            site, group, pid, _ = selected_listener_path.split('/')

            # Always use the first video of the target individual for person-specific behaviour pattern learning
            # listener_emotion_path = os.path.join(self._emotion_path, '/'.join([site, group, pid, '1'])+'.csv')
            listener_emotion_path = os.path.join(self._emotion_path, '/'.join([site, group, pid, '1'])+'.csv')
            
            if 'NoXI' in listener_emotion_path:
                listener_emotion_path=listener_emotion_path.replace('Novice_video','P2')
                listener_emotion_path=listener_emotion_path.replace('Expert_video','P1')

            if 'Emotion/RECOLA/group' in listener_emotion_path:
                listener_emotion_path=listener_emotion_path.replace('P25','P1')
                listener_emotion_path=listener_emotion_path.replace('P26','P2')
                listener_emotion_path=listener_emotion_path.replace('P41','P1')
                listener_emotion_path=listener_emotion_path.replace('P42','P2')
                listener_emotion_path=listener_emotion_path.replace('P45','P1')
                listener_emotion_path=listener_emotion_path.replace('P46','P2')

            listener_emotion_path = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            # shape: [_personal_clip_length, 25]
            listener_personal_clip = torch.from_numpy(np.array(listener_emotion_path.drop(0)).astype(np.float32))[:self._personal_clip_length] 
            
        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio:
            speaker_audio_path = os.path.join(self._audio_path, self.speaker_path[index]+'.wav')
            speaker_audio_clip = extract_audio_features(speaker_audio_path, self._fps, total_length)

            # shape: [_clip_length, 78]
            speaker_audio_clip = torch.from_numpy(speaker_audio_clip[cp:cp + self._clip_length])
            
        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, inappro_listener_emotion, speaker_emotion = 0, 0, 0

        if self.load_emotion_l:
            selected_listener_emotion = []
            union_index = np.concatenate([appro_listeners_index, inappro_listeners_index])

            for i in union_index:
                listener_emotion_path = os.path.join(self._emotion_path, self.listener_path[i]+'.csv')

                if 'NoXI' in listener_emotion_path:
                    listener_emotion_path=listener_emotion_path.replace('Novice_video','P2')
                    listener_emotion_path=listener_emotion_path.replace('Expert_video','P1')

                if 'Emotion/RECOLA/group' in listener_emotion_path:
                    listener_emotion_path=listener_emotion_path.replace('P25','P1')
                    listener_emotion_path=listener_emotion_path.replace('P26','P2')
                    listener_emotion_path=listener_emotion_path.replace('P41','P1')
                    listener_emotion_path=listener_emotion_path.replace('P42','P2')
                    listener_emotion_path=listener_emotion_path.replace('P45','P1')
                    listener_emotion_path=listener_emotion_path.replace('P46','P2')

                listener_emotion_path = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
                listener_emotion_path = torch.from_numpy(np.array(listener_emotion_path.drop(0)).astype(np.float32))[
                    cp: cp + self._clip_length]
                
                selected_listener_emotion.append(listener_emotion_path)

            listener_emotion = selected_listener_emotion[:appro_listeners_index.shape[0]]
            inappro_listener_emotion = selected_listener_emotion[appro_listeners_index.shape[0]:]

            # shape: [len(appro_listeners_index), _clip_length, 25]
            listener_emotion = torch.stack(listener_emotion)

            if len(inappro_listener_emotion) != 0:
                # shape: [len(inappro_listeners_index), _clip_length, 25]
                inappro_listener_emotion = torch.stack(inappro_listener_emotion)

        if self.load_emotion_s:
            speaker_emotion_path = os.path.join(self._emotion_path, self.speaker_path[index]+'.csv')

            if 'NoXI' in speaker_emotion_path:
                speaker_emotion_path=speaker_emotion_path.replace('Novice_video','P2')
                speaker_emotion_path=speaker_emotion_path.replace('Expert_video','P1')

            if 'Emotion/RECOLA/group' in speaker_emotion_path:
                speaker_emotion_path=speaker_emotion_path.replace('P25','P1')
                speaker_emotion_path=speaker_emotion_path.replace('P26','P2')
                speaker_emotion_path=speaker_emotion_path.replace('P41','P1')
                speaker_emotion_path=speaker_emotion_path.replace('P42','P2')
                speaker_emotion_path=speaker_emotion_path.replace('P45','P1')
                speaker_emotion_path=speaker_emotion_path.replace('P46','P2')

            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')

            # shape: [1, 25]
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                cp: cp + self._clip_length]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm, inappro_listener_3dmm = 0, 0
        if self.load_3dmm_l:
            selected_listener_3dmm = []
            union_index = np.concatenate([appro_listeners_index, inappro_listeners_index])
            
            for i in union_index:
                listener_3dmm_path = os.path.join(self._3dmm_path, self.listener_path[i]+'.npy')
                listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
                listener_3dmm = listener_3dmm[cp: cp + self._clip_length]
                listener_3dmm = self._transform_3dmm(listener_3dmm)[0]

                selected_listener_3dmm.append(listener_3dmm)

            listener_3dmm = selected_listener_3dmm[:appro_listeners_index.shape[0]]
            inappro_listener_3dmm = selected_listener_3dmm[appro_listeners_index.shape[0]:]

            # shape: [len(appro_listeners_index), _clip_length, 25]
            listener_3dmm = torch.stack(listener_3dmm)

            if len(inappro_listener_3dmm) != 0:
                # shape: [len(inappro_listeners_index), _clip_length, 25]
                inappro_listener_3dmm = torch.stack(inappro_listener_3dmm)

        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = os.path.join(self._3dmm_path, self.listener_path[index]+'.npy')
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[cp: cp + self._clip_length]
            speaker_3dmm = self._transform_3dmm(speaker_3dmm)[0]

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:
            selected_listener_index = selected_listener_index if self._stage == 'person_specific' else index
            listener_video_path = os.path.join(self._video_path, self.listener_path[selected_listener_index]+'.mp4')
            container = av.open(listener_video_path)  #read mp4 files

            for frame in container.decode(video=0):
                img=frame.to_image().convert('RGB')
                # img = self._transform(img)
                break

            # shape: [3, 224, 224]
            listener_reference = self._transform(img)

        # ====================== Number of Appropriate Reactions ======================
        num_appropriate = len(appro_listeners_index)
        num_inappropriate = len(inappro_listeners_index)

        return speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_personal_clip, listener_audio_clip, listener_emotion, inappro_listener_emotion, listener_3dmm, inappro_listener_3dmm, listener_reference, num_appropriate, num_inappropriate

    def __len__(self):
        return self._len
    
def get_dataloader(conf, split, stage, load_audio=False, load_video_s=False, load_video_l=False, load_clip_p=False, load_emotion_s=False,
                   load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=True):

    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    print('==> Preparing data for {}...'.format(split) + '\n')

    dataset = ReactionDataset(root_path=conf.dataset_path, split=split, stage=stage, img_size=conf.img_size, crop_size=conf.crop_size,
                              clip_length=conf.clip_length, personal_clip_length=720, fps=25,
                              load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l, load_clip_p=load_clip_p,
                              load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
                              load_3dmm_l=load_3dmm_l, load_ref=load_ref, repeat_mirrored=repeat_mirrored)

    shuffle = True if split == "train" else False
    dataloader = DataLoader(dataset=dataset, collate_fn=custom_collate, batch_size=conf.batch_size, shuffle=shuffle, num_workers=conf.num_workers)
    
    return dataloader