import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# def rescale_time(time_array):
#   time_array = (time_array - np.mean(time_array))/np.std(time_array)
#   return time_array

# class QuestionStatic(nn.Module):


class UserTemp(nn.Module):
    def __init__(self, embed_dim, hidden_dim, questionset_size, tagset_size, base):
        super(UserTemp, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.base = base

        print("Base model: ", self.base)

        q_padding_idx = questionset_size
        tag_padding_idx = tagset_size

        self.embed_q = nn.Embedding(questionset_size+1, self.embed_dim, padding_idx=q_padding_idx)
        self.embed_t = nn.Embedding(tagset_size+1, self.embed_dim, padding_idx=tag_padding_idx)
        # self.embed = nn.Linear(embed_dim + 3, embed_dim, bias=False) 

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # self.lstm = nn.LSTM(input_size=embed_dim*2 + 3, hidden_size=hidden_dim, num_layers=1) # +3 for newbie, timestep_prev, timestep
        self.lstm = nn.LSTM(input_size=embed_dim*2 + 1, hidden_size=hidden_dim, num_layers=1) # +1 for timestep gap
        self.output = nn.Linear(hidden_dim, 1)

        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, questions, timestamps, tags):
        timestamps = timestamps.unsqueeze(2)
        # prev_timestamps = timestamps.clone()
        # prev_timestamps[:,1:] = timestamps[:,:-1]
        # prev_timestamps[:,0] = timestamps[:,0]
        gaps = torch.zeros_like(timestamps)
        gaps[:,1:] = timestamps[:,1:] - timestamps[:,:-1] #There will be a negative where the padding starts, but I don't think it affects learning

        # newbie = torch.zeros_like(timestamps)
        # newbie[:,0] = 1                   # TEMPORARY, CHANGE FOR BROKEN-UP SEQUENCES LATER

        embed_qs = self.embed_q(questions)
        embed_ts = self.embed_t(tags)

        embed_ts = torch.sum(embed_ts, dim=2)

        # embeds = torch.cat((embed_qs, embed_ts, newbie, prev_timestamps, timestamps), dim=2)
        # embeds = torch.cat((embed_qs, embed_ts, newbie, gaps), dim=2)
        embeds = torch.cat((embed_qs, embed_ts, gaps), dim=2)

        lstm_out, ____ = self.lstm(embeds.float())

        user_temp_contribution = self.output(lstm_out).squeeze()

        if self.base:
            out = torch.tanh(user_temp_contribution)
        else:
            out = torch.sigmoid(user_temp_contribution)
        return out

    def forward_combined(self, questions, timestamps, tags, base_model_outputs):
        timestamps = timestamps.unsqueeze(2)

        gaps = torch.zeros_like(timestamps)
        gaps[:,1:] = timestamps[:,1:] - timestamps[:,:-1] 

        print(gaps[0])

        embed_qs = self.embed_q(questions)
        embed_ts = self.embed_t(tags)

        embed_ts = torch.sum(embed_ts, dim=2)
        embeds = torch.cat((embed_qs, embed_ts, gaps), dim=2)

        lstm_out, ____ = self.lstm(embeds.float())

        user_temp_contribution = self.output(lstm_out).squeeze()
        
        out = torch.sigmoid(user_temp_contribution + base_model_outputs)

        return out



# class CombinedModel(nn.Module):


# class RRNDataset(Dataset):
#     """RRN dataset."""

#     def __init__(self, data_path, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         with open(data_path, 'rb') as f:
#             self.data = pickle.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#             sample = self.data[self.data['user'] in idx]
#         else:
#             sample = self.data[self.data['user'] == idx]

#         return sample
