import torch
import torch.nn as nn
import torch.nn.functional as F
from model.TransformerVAE import TransformerVAE, VideoEncoder
from model.BasicBlock import PositionalEncoding, CrossAttn, init_biased_mask

class PFE(nn.Module):
    def __init__(self, emotion_dim = 25, feature_dim = 128, k=2):
        super(PFE, self).__init__()

        self.feature_dim = feature_dim
        self.emotion_dim = emotion_dim
        # self.clip_unit = clip_unit
        self.k = k

        if self.k == 0:
            self.clip_unit = 480
        elif self.k == 1:
            self.clip_unit = 240
        elif self.k == 2:
            self.clip_unit = 120
        elif self.k == 3:
            self.clip_unit = 60
        elif self.k == 4:
            self.clip_unit = 30

        self.listener_behaviour_encoder = nn.Sequential(
            nn.Linear(emotion_dim, feature_dim*2),
            nn.Linear(feature_dim*2, feature_dim)
        )

        self.cross_attn = CrossAttn(in_channels=feature_dim)

        self.temporal_layer = nn.Sequential(
            nn.Linear(self.clip_unit * feature_dim, feature_dim*2),
            nn.Linear(feature_dim*2, feature_dim)
        )

    def forward(self, listener_personal_clip=None):
        # shape: bs, clip_unit*4, 25
        B, T, D = listener_personal_clip.shape

        # print('listener_personal_clip shape:', listener_personal_clip.shape)  # torch.Size([4, 720, 25])

        encoded_feature = self.listener_behaviour_encoder(listener_personal_clip)
        # print('encoded_feature shape before cross_attn:', encoded_feature.shape)  # torch.Size([4, 720, 128])


        if self.k == 1:
            encoded_feature = self.cross_attn(encoded_feature[:,0:self.clip_unit,:], encoded_feature[:,self.clip_unit:2*self.clip_unit,:])
		
            encoded_feature = encoded_feature.view(B, -1)
            person_specific_factor = self.temporal_layer(encoded_feature)
            
            return person_specific_factor
        elif self.k == 2:
            encoded_feature_1 = self.cross_attn(encoded_feature[:,0:self.clip_unit,:], encoded_feature[:,2*self.clip_unit:3*self.clip_unit,:])
            encoded_feature_2 = self.cross_attn(encoded_feature[:,self.clip_unit:2*self.clip_unit,:], encoded_feature[:,3*self.clip_unit:4*self.clip_unit,:])

            encoded_feature = self.cross_attn(encoded_feature_1, encoded_feature_2)

            encoded_feature = encoded_feature.view(B, -1)
            person_specific_factor = self.temporal_layer(encoded_feature)
            
            return person_specific_factor
        elif self.k == 3:
            encoded_feature_1 = self.cross_attn(encoded_feature[:,0:self.clip_unit,:], encoded_feature[:,4*self.clip_unit:5*self.clip_unit,:])
            encoded_feature_2 = self.cross_attn(encoded_feature[:,self.clip_unit:2*self.clip_unit,:], encoded_feature[:,5*self.clip_unit:6*self.clip_unit,:])
            encoded_feature_3 = self.cross_attn(encoded_feature[:,2*self.clip_unit:3*self.clip_unit,:], encoded_feature[:,6*self.clip_unit:7*self.clip_unit,:])
            encoded_feature_4 = self.cross_attn(encoded_feature[:,3*self.clip_unit:4*self.clip_unit,:], encoded_feature[:,7*self.clip_unit:8*self.clip_unit,:])

            encoded_feature_1 = self.cross_attn(encoded_feature_1, encoded_feature_3)
            encoded_feature_2 = self.cross_attn(encoded_feature_2, encoded_feature_4)

            encoded_feature = self.cross_attn(encoded_feature_1, encoded_feature_2)

            encoded_feature = encoded_feature.view(B, -1)
            person_specific_factor = self.temporal_layer(encoded_feature)
            
            return person_specific_factor
        
        elif self.k == 4:
            # Split the clip into four unit clips
            encoded_feature_1 = self.cross_attn(encoded_feature[:,0:self.clip_unit,:], encoded_feature[:,8*self.clip_unit:9*self.clip_unit,:])
            encoded_feature_2 = self.cross_attn(encoded_feature[:,self.clip_unit:2*self.clip_unit,:], encoded_feature[:,9*self.clip_unit:10*self.clip_unit,:])
            encoded_feature_3 = self.cross_attn(encoded_feature[:,2*self.clip_unit:3*self.clip_unit,:], encoded_feature[:,10*self.clip_unit:11*self.clip_unit,:])
            encoded_feature_4 = self.cross_attn(encoded_feature[:,3*self.clip_unit:4*self.clip_unit,:], encoded_feature[:,11*self.clip_unit:12*self.clip_unit,:])
            encoded_feature_5 = self.cross_attn(encoded_feature[:,4*self.clip_unit:5*self.clip_unit,:], encoded_feature[:,12*self.clip_unit:13*self.clip_unit,:])
            encoded_feature_6 = self.cross_attn(encoded_feature[:,5*self.clip_unit:6*self.clip_unit,:], encoded_feature[:,13*self.clip_unit:14*self.clip_unit,:])
            encoded_feature_7 = self.cross_attn(encoded_feature[:,6*self.clip_unit:7*self.clip_unit,:], encoded_feature[:,14*self.clip_unit:15*self.clip_unit,:])
            encoded_feature_8 = self.cross_attn(encoded_feature[:,7*self.clip_unit:8*self.clip_unit,:], encoded_feature[:,15*self.clip_unit:16*self.clip_unit,:])


            encoded_feature_1 = self.cross_attn(encoded_feature_1, encoded_feature_5)
            encoded_feature_2 = self.cross_attn(encoded_feature_2, encoded_feature_6)
            encoded_feature_3 = self.cross_attn(encoded_feature_3, encoded_feature_7)
            encoded_feature_4 = self.cross_attn(encoded_feature_4, encoded_feature_8)

            encoded_feature_1 = self.cross_attn(encoded_feature_1, encoded_feature_3)
            encoded_feature_2 = self.cross_attn(encoded_feature_2, encoded_feature_4)

            encoded_feature = self.cross_attn(encoded_feature_1, encoded_feature_2)

            encoded_feature = encoded_feature.view(B, -1)
            person_specific_factor = self.temporal_layer(encoded_feature)
            
            return person_specific_factor


class PFM(nn.Module):
    def __init__(self, img_size=224, audio_dim = 78, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, seq_len=751, general_branch_path='', integration = None, device = 'cpu', k=2, beta=0.1):
        super(PFM, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.general_branch_path = general_branch_path
        self.beta = beta
        
        self.general_branch = TransformerVAE(img_size = img_size, audio_dim = audio_dim,  output_3dmm_dim = output_3dmm_dim, output_emotion_dim = output_emotion_dim, feature_dim = feature_dim, seq_len = seq_len, integration = integration, device = device, beta=self.beta)
        
        if self.general_branch_path != '':
            self.load_general_branch()

        # Only freeze the encoder of GFRG and reuse the decoder for further training
        for name, module in self.general_branch.named_modules():
            if 'speaker_behaviour_encoder' in name or 'vae_model' in name:
                for param in module.parameters():
                    param.requires_grad = False

        self.person_branch = PFE(emotion_dim = output_emotion_dim, feature_dim = feature_dim, k=k)

    def load_general_branch(self):
        checkpoint_path = self.general_branch_path
        print("Load the general branch from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        self.general_branch.load_state_dict(state_dict, strict=False)

    def forward(self, speaker_video=None, speaker_audio=None, listener_personal_clip=None, num_samples=None, **kwargs):

        """
        input:
        video: (batch_size, seq_len, 3, img_size, img_size)
        audio: (batch_size, raw_wav)

        output:
        3dmm_vector: (batch_size, seq_len, output_3dmm_dim)
        emotion_vector: (batch_size, seq_len, output_emotion_dim)
        """
        person_specific_factor = self.person_branch(listener_personal_clip)
        listener_3dmm_out, listener_emotion_out, _, _ = self.general_branch(speaker_video=speaker_video, speaker_audio=speaker_audio, num_samples=num_samples, person_specific_factor=person_specific_factor)

        return listener_3dmm_out, listener_emotion_out, person_specific_factor