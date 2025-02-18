import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicBlock import CrossAttn

class Appropriateness_Discriminator(nn.Module):
    def __init__(self, emotion_dim = 25, _3dmm_dim = 58, feature_dim = 128, output_dim = 1):
        super(Appropriateness_Discriminator, self).__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self._3dmm_encoder = nn.Linear(_3dmm_dim, feature_dim)
        self._emotion_encoder = nn.Linear(emotion_dim, feature_dim)
        
        self.cross_attn_emotion = CrossAttn(feature_dim)
        self.cross_attn_3dmm = CrossAttn(feature_dim)

        self.fusion = nn.Linear(2*self.feature_dim, self.feature_dim)
        
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=3, batch_first=True)

        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, speaker_emotion=None, listener_emotion=None, speaker_3dmm=None, listener_3dmm=None, repeat_interleave=None, person_specific_factor=None, p_weight=0.00001):
        speaker_emotion = speaker_emotion.repeat_interleave(repeat_interleave, dim=0)
        speaker_3dmm = speaker_3dmm.repeat_interleave(repeat_interleave, dim=0)
        if person_specific_factor != None:
            person_specific_factor = person_specific_factor.repeat_interleave(repeat_interleave, dim=0).unsqueeze(1)

        listener_emotion = self._emotion_encoder(listener_emotion)
        speaker_emotion = self._emotion_encoder(speaker_emotion)
        if person_specific_factor != None:
            speaker_emotion = torch.cat([p_weight * person_specific_factor, speaker_emotion], dim=1)
        _emotion_feature = self.cross_attn_emotion(listener_emotion, speaker_emotion)

        listener_3dmm = self._3dmm_encoder(listener_3dmm)
        speaker_3dmm = self._3dmm_encoder(speaker_3dmm)
        if person_specific_factor != None:
            speaker_3dmm = torch.cat([p_weight * person_specific_factor, speaker_3dmm], dim=1)
        _3dmm_feature = self.cross_attn_3dmm(listener_3dmm, speaker_3dmm)

        encoded_feature = self.fusion(torch.cat((_emotion_feature, _3dmm_feature), dim=-1))
        
        hidden = None
        for t in range(encoded_feature.size(1)):
            x = encoded_feature[:, t, :]
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        out = self.fc1(out[-1, :, :])
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.output_dim == 1:
            out = torch.sigmoid(out)
        
        return out

# Implement IM as concatenation
class Discriminator_IM_Cat(nn.Module):
    def __init__(self, emotion_dim = 25, _3dmm_dim = 58, feature_dim = 128, output_dim = 1):
        super(Discriminator_IM_Cat, self).__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self._3dmm_encoder = nn.Linear(_3dmm_dim, feature_dim)
        self._emotion_encoder = nn.Linear(emotion_dim, feature_dim)
        
        self._3dmm_fusion = nn.Linear(2*self.feature_dim, feature_dim)
        self._emotion_fusion = nn.Linear(2*self.feature_dim, feature_dim)

        self.fusion = nn.Linear(2*self.feature_dim, self.feature_dim)
        
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=3, batch_first=True)

        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, speaker_emotion=None, listener_emotion=None, speaker_3dmm=None, listener_3dmm=None, repeat_interleave=None, **kwargs):
        speaker_emotion = speaker_emotion.repeat_interleave(repeat_interleave, dim=0)
        speaker_3dmm = speaker_3dmm.repeat_interleave(repeat_interleave, dim=0)

        listener_emotion = self._emotion_encoder(listener_emotion)
        speaker_emotion = self._emotion_encoder(speaker_emotion)
        _emotion_feature = self._emotion_fusion(torch.cat([listener_emotion, speaker_emotion], dim=-1))
        

        listener_3dmm = self._3dmm_encoder(listener_3dmm)
        speaker_3dmm = self._3dmm_encoder(speaker_3dmm)
        _3dmm_feature = self._3dmm_fusion(torch.cat([listener_3dmm, speaker_3dmm], dim=-1))

        encoded_feature = self.fusion(torch.cat((_emotion_feature, _3dmm_feature), dim=-1))
        
        hidden = None
        for t in range(encoded_feature.size(1)):
            x = encoded_feature[:, t, :]
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        out = self.fc1(out[-1, :, :])
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.output_dim == 1:
            out = torch.sigmoid(out)
        
        return out
    
class Discriminator_IM_Sum(nn.Module):
    def __init__(self, emotion_dim = 25, _3dmm_dim = 58, feature_dim = 128, output_dim = 1):
        super(Discriminator_IM_Sum, self).__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self._3dmm_encoder = nn.Linear(_3dmm_dim, feature_dim)
        self._emotion_encoder = nn.Linear(emotion_dim, feature_dim)
        
        self._3dmm_fusion = nn.Linear(self.feature_dim, feature_dim)
        self._emotion_fusion = nn.Linear(self.feature_dim, feature_dim)

        self.fusion = nn.Linear(2*self.feature_dim, self.feature_dim)
        
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, num_layers=3, batch_first=True)

        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, speaker_emotion=None, listener_emotion=None, speaker_3dmm=None, listener_3dmm=None, repeat_interleave=None, **kwargs):
        speaker_emotion = speaker_emotion.repeat_interleave(repeat_interleave, dim=0)
        speaker_3dmm = speaker_3dmm.repeat_interleave(repeat_interleave, dim=0)

        listener_emotion = self._emotion_encoder(listener_emotion)
        speaker_emotion = self._emotion_encoder(speaker_emotion)
        _emotion_feature = listener_emotion + speaker_emotion
        

        listener_3dmm = self._3dmm_encoder(listener_3dmm)
        speaker_3dmm = self._3dmm_encoder(speaker_3dmm)
        _3dmm_feature = listener_3dmm + speaker_3dmm

        encoded_feature = self.fusion(torch.cat((_emotion_feature, _3dmm_feature), dim=-1))
        
        hidden = None
        for t in range(encoded_feature.size(1)):
            x = encoded_feature[:, t, :]
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        out = self.fc1(out[-1, :, :])
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.output_dim == 1:
            out = torch.sigmoid(out)
        
        return out
    
class Frame_Discrminator(nn.Module):
    def __init__(self, feature_dim = 58, latent_dim=128, output_dim = 1):
        super(Frame_Discrminator, self).__init__()

        self.output_dim = 1

        self.fc1 = nn.Linear(feature_dim, latent_dim)  # First fully connected layer
        self.fc2 = nn.Linear(latent_dim, latent_dim*2) # Second fully connected layer
        self.fc3 = nn.Linear(latent_dim*2, latent_dim*4) # Third fully connected layer
        self.fc4 = nn.Linear(latent_dim*4, output_dim)   # Output layer

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)  # Leaky ReLU activation with negative slope of 0.2
        x = F.dropout(x, 0.3)  # Dropout for regularization
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = self.fc4(x)

        if self.output_dim == 1:
            x = torch.sigmoid(x)  # Sigmoid activation for binary classification output

        return x