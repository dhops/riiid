import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def rescale_time(time_array):
  time_array = (time_array - np.mean(time_array))/np.std(time_array)
  return time_array

class QuestionStatic(nn.Module):


class QuestionTemp(nn.Module):

    def __init__(self, embedding_dim=40, hidden_dim=40, questionset_size, tagset_size ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.embed = nn.Linear(questionset_size + 3, embedding_dim, bias=False) # +3 for newbie, timestep_prev, timestep

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, questions):
        embeds = self.embed(questions)
        lstm_out, lstm_hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return lstm_hidden

