import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.BasicBlock import ConvBlock, PositionalEncoding, CrossAttn, lengths_to_mask, init_biased_mask

class VideoEncoder(nn.Module):
    def __init__(self, img_size=224, feature_dim = 128, device = 'cpu'):
        super(VideoEncoder, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim

        self.Conv3D = ConvBlock(3, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.device = device


    def forward(self, video):
        """
        input:
        speaker_video_frames x: (batch_size, seq_len, 3, img_size, img_size)

        output:
        speaker_temporal_tokens y: (batch_size, seq_len, token_dim)

        """

        video_input = video.transpose(1, 2)  # B C T H W
        token_output = self.Conv3D(video_input).transpose(1,2)
        token_output = self.fc(token_output) # B T C
        return  token_output
    
class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self, img_size=224, audio_dim = 78, feature_dim = 128,  device = 'cpu'):
        super(SpeakerBehaviourEncoder, self).__init__()

        self.img_size = img_size
        self.audio_dim = audio_dim
        self.feature_dim = feature_dim
        self.device = device

        self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
        self.audio_feature_map = nn.Linear(self.audio_dim, self.feature_dim)
        self.fusion_layer = nn.Linear(self.feature_dim*2, self.feature_dim)


    def forward(self, video, audio):
        video_feature = self.video_encoder(video)
        
        audio_feature = self.audio_feature_map(audio)
        
        speaker_behaviour_feature = self.fusion_layer(torch.cat((video_feature, audio_feature), dim =-1))

        return  speaker_behaviour_feature
    
class VAEModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 256,
                 **kwargs) -> None:
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))


    def forward(self, input, num_samples):
        x = self.linear(input)  # B T D
        # print('VAEModel - x shape:', x.shape)
        B, T, D = input.shape   # 4, 256, 128

        lengths = [len(item) for item in input] #[256, 256, 256, 256]

        mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)    # shape torch.Size([4, 1, 128])
        logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)    # shape torch.Size([4, 1, 128])

        x = torch.cat([mu_token, logvar_token, x], dim=1)   # shape torch.Size([4, 258, 128])

        x = x.permute(1, 0, 2)     # shape torch.Size([258, 4, 128])

        token_mask = torch.ones((B, 2), dtype=bool, device=input.get_device())
        mask = lengths_to_mask(lengths, input.get_device())

        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask) # shape torch.Size([258, 4, 128])
        
        mu = x[0].repeat_interleave(num_samples, dim=0)
        logvar = x[1].repeat_interleave(num_samples, dim=0)
        std = logvar.exp().pow(0.5)

        dist = torch.distributions.Normal(mu, std)
        motion_sample = self.sample_from_distribution(dist).to(input.get_device())
        # return motion_sample, dist
        return motion_sample, torch.distributions.Normal(x[0], x[1].exp().pow(0.5))

    def sample_from_distribution(self, distribution):
         return distribution.rsample()
    
class Decoder(nn.Module):
    def __init__(self,  output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, device = 'cpu', seq_len=256, max_seq_len=751, n_head = 4, integration = None, beta = 0.1):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.device = device
        self.integration = integration
        self.beta = beta

        self.vae_model = VAEModel(feature_dim, feature_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = max_seq_len, period=max_seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )

        if self.integration == 'summation_beta':
            self.raw_beta = nn.Parameter(data=torch.Tensor(1), requires_grad=True)

        if self.integration == 'concat':
            self.concat = nn.Linear(feature_dim*2, feature_dim)

        # Decoder Layer for Integration
        if self.integration == 'self-attn' or self.integration == 'residual' :
            self.person_specific_reaction_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.PE = PositionalEncoding(feature_dim)

    def forward(self, encoded_feature, person_specific_factor = None, num_samples = None):
        # encoded_feature shape: B, T, D
        B, TL = encoded_feature.shape[0], encoded_feature.shape[1]
        
        # motion_sample shape: [num_samples.sum(), D]
        motion_sample, dist = self.vae_model(encoded_feature, num_samples=num_samples)

        # Summation
        if self.integration == 'summation' and person_specific_factor != None:
            motion_sample += self.beta * person_specific_factor.repeat_interleave(num_samples, dim=0)

        # Summation based on beta 
        if self.integration == 'summation_beta' and person_specific_factor != None:
            motion_sample += torch.sigmoid(self.raw_beta) * person_specific_factor.repeat_interleave(num_samples, dim=0)

        # Concatenation
        if self.integration == 'concat':
            motion_sample = self.concat(torch.cat([motion_sample, person_specific_factor.repeat_interleave(num_samples, dim=0)], dim=-1))

        time_queries = torch.zeros(num_samples.sum(), TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().to(device=self.device).repeat(num_samples.sum(),1,1)

        listener_reaction = self.listener_reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1), tgt_mask=tgt_mask)

        if (self.integration == 'self-attn' or self.integration == 'residual') and person_specific_factor != None:
            ptgt_mask = self.biased_mask[:, :TL+1, :TL+1].clone().detach().to(device=self.device).repeat(num_samples.sum(),1,1)

            x = torch.cat([person_specific_factor.repeat_interleave(num_samples, dim=0).unsqueeze(1), listener_reaction], dim=1)
            x = self.person_specific_reaction_decoder(x, x, tgt_mask=ptgt_mask)

            if self.integration == 'self-attn':
                listener_reaction = x[:, 1:TL+1, :]
            elif self.integration == 'residual':
                listener_reaction += x[:, 1:TL+1, :]

        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction, tgt_mask=tgt_mask)

        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)

        listener_emotion_out = self.listener_reaction_emotion_map_layer(
            torch.cat((listener_3dmm_out, listener_reaction), dim=-1))

        return listener_3dmm_out, listener_emotion_out, motion_sample, dist
    
class TransformerVAE(nn.Module):
    def __init__(self, img_size=224, audio_dim = 78, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, seq_len=751, integration=None, device = 'cpu', beta=0.1):
        super(TransformerVAE, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.integration = integration
        self.beta = beta

        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(img_size, audio_dim, feature_dim, device)
        self.reaction_decoder = Decoder(output_3dmm_dim = output_3dmm_dim, output_emotion_dim = output_emotion_dim, feature_dim = feature_dim,  device=device, seq_len=self.seq_len, integration=self.integration, beta=self.beta)
        self.fusion = nn.Linear(feature_dim + self.output_3dmm_dim + self.output_emotion_dim, feature_dim)

    def forward(self, speaker_video=None, speaker_audio=None, num_samples=None, person_specific_factor=None,  **kwargs):

        """
        input:
        video: (batch_size, seq_len, 3, img_size, img_size)
        audio: (batch_size, raw_wav)

        output:
        3dmm_vector: (batch_size, seq_len, output_3dmm_dim)
        emotion_vector: (batch_size, seq_len, output_emotion_dim)
        distribution: [dist_1,...,dist_n]
        """
        distribution = []

        encoded_feature = self.speaker_behaviour_encoder(speaker_video, speaker_audio)
        listener_3dmm_out, listener_emotion_out, motion_sample, dist = self.reaction_decoder(encoded_feature=encoded_feature, person_specific_factor=person_specific_factor, num_samples=num_samples)
        distribution.append(dist)
        
        listener_emotion_out[:, :, :15] = torch.sigmoid(listener_emotion_out[:, :, :15])
        listener_emotion_out[:, :, 17:] = torch.softmax(listener_emotion_out[:, :, 17:], dim=2)

        return listener_3dmm_out, listener_emotion_out, motion_sample, distribution