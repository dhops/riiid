import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class RRNCF(torch.nn.Module):
    """
    A Pytorch implementation of Neural Collaborative Filtering.

    Reference:
        X He, et al. Neural Collaborative Filtering, 2017.
    """

    def __init__(self, embed_dim, mlp_dim, dropout, questionset_size, tagset_size, userset_size):
        super().__init__()
        self.q_embed_dim = 12
        self.mlp_dim = 16
        self.lstm_dim = 32
        self.t_embed_dim = 4

        # self.user_embedding = torch.nn.Embedding(userset_size+1, self.embed_dim, padding_idx=userset_size)
        self.question_embedding = torch.nn.Embedding(questionset_size+1, self.q_embed_dim, padding_idx=questionset_size)
        self.tag_embedding = torch.nn.Embedding(tagset_size+1, self.t_embed_dim, padding_idx=tagset_size)

        self.embed_output_dim = self.q_embed_dim + self.t_embed_dim

        self.lstm = torch.nn.LSTM(input_size=self.embed_output_dim + 2, batch_first=True, hidden_size=self.lstm_dim, num_layers=1) # 2 for gaps and corrects


        self.mlp = torch.nn.Sequential(
          torch.nn.Linear(self.lstm_dim + self.q_embed_dim + self.t_embed_dim, mlp_dim),
          torch.nn.BatchNorm1d(mlp_dim),
          torch.nn.ReLU(),
          torch.nn.Dropout(p=dropout),
          torch.nn.Linear(mlp_dim, mlp_dim),
          torch.nn.BatchNorm1d(mlp_dim),
          torch.nn.ReLU(),
          torch.nn.Dropout(p=dropout),
        )
        # self.fc = torch.nn.Linear(self.mlp_dim + self.lstm_dim, 1)

        #try without GMF
        self.fc = torch.nn.Linear(self.mlp_dim, 1)

    def forward(self, questions, timestamps, tags, targets, users, q_lens):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        # timestamps = timestamps
        # gaps = torch.zeros_like(timestamps).float()
        # gaps[:,1:] = timestamps[:,1:] - timestamps[:,:-1] #There will be a negative where the padding starts, but I don't think it affects learning

        #normalize batch of gaps (SLOPPY, PLEASE FIX!)
        # std = torch.std(gaps)
        # mean = torch.mean(gaps)
        # gaps = (gaps - gaps.mean())/gaps.std()

        gaps = timestamps.unsqueeze(2)

        # timestamps --> (normalized) gaps in data module
        # gaps = timestamps.unsqueeze(2)

        embed_qs = self.question_embedding(questions)
        embed_ts = self.tag_embedding(tags)
        embed_ts = torch.sum(embed_ts, dim=2)

        targets = targets.unsqueeze(2)

        lstm_embeds = torch.cat((embed_qs, embed_ts, gaps, targets), dim=2).float()
        lstm_in_packed = pack_padded_sequence(lstm_embeds, q_lens, batch_first=True, enforce_sorted=False)

        lstm_out_packed, ____ = self.lstm(lstm_in_packed)
        lstm_out, output_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        user_knowledge = torch.roll(lstm_out, 1, dims=1)
        user_knowledge[:,0] = 0.0

        user_knowledge = torch.reshape(user_knowledge, (-1, self.lstm_dim))
        embed_qs = torch.reshape(embed_qs, (-1, self.q_embed_dim))

        # users = users.unsqueeze(1)
        # users = users.repeat(1,questions.shape[1])
        # embed_us = self.user_embedding(users)
        # embed_us = torch.reshape(embed_us, (-1, self.embed_dim))

        embed_ts = torch.reshape(embed_ts, (-1, self.t_embed_dim))

        x = torch.cat((user_knowledge, embed_qs, embed_ts), dim=1)
        x = self.mlp(x)

        # gmf = user_knowledge * torch.cat((embed_qs, embed_ts), dim=1)

        # x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)

        return torch.sigmoid(x)

    def forward_tbptt(self, questions, timestamps, tags, targets, users, q_lens, h_t_c_t=None):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        gaps = timestamps.unsqueeze(2)
        embed_qs = self.question_embedding(questions)
        embed_ts = self.tag_embedding(tags)
        embed_ts = torch.sum(embed_ts, dim=2)

        targets = targets.unsqueeze(2)

        lstm_embeds = torch.cat((embed_qs, embed_ts, gaps, targets), dim=2).float()
        # lstm_in_packed = pack_padded_sequence(lstm_embeds, q_lens, batch_first=True, enforce_sorted=False)

        if h_t_c_t is not None:
            lstm_out, h_t_c_t = self.lstm(lstm_embeds, h_t_c_t)
        else:
            lstm_out, h_t_c_t = self.lstm(lstm_embeds)
        # lstm_out, output_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)

        user_knowledge = torch.roll(lstm_out, 1, dims=1)
        user_knowledge[:,0] = 0.0

        user_knowledge = torch.reshape(user_knowledge, (-1, self.lstm_dim))
        embed_qs = torch.reshape(embed_qs, (-1, self.q_embed_dim))

        embed_ts = torch.reshape(embed_ts, (-1, self.t_embed_dim))

        x = torch.cat((user_knowledge, embed_qs, embed_ts), dim=1)
        x = self.mlp(x)

        # gmf = user_knowledge * torch.cat((embed_qs, embed_ts), dim=1)

        # x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)

        return torch.sigmoid(x), h_t_c_t


    def get_user_state(self, questions, timestamps, tags, targets, verbose=False):
        gaps = timestamps.unsqueeze(2)
        embed_qs = self.question_embedding(questions)
        embed_ts = self.tag_embedding(tags)
        embed_ts = torch.sum(embed_ts, dim=2)

        targets = targets.unsqueeze(2)

        lstm_embeds = torch.cat((embed_qs, embed_ts, gaps, targets), dim=2)
        # TRY WITH NO GAPS (Then normalize gaps)
        # lstm_embeds = torch.cat((embed_qs, embed_ts, targets), dim=2)
        lstm_out, ____ = self.lstm(lstm_embeds.float())

        user_knowledge = torch.roll(lstm_out, 1, dims=1)
        user_knowledge[:,0] = 0.0

        return user_knowledge


    # def forward_no_sigmoid(self, x):
    #     """
    #     :param x: Long tensor of size ``(batch_size, num_user_fields)``
    #     """
    #     x = self.embedding(x)
    #     user_x = x[:, self.user_field_idx].squeeze(1)
    #     item_x = x[:, self.item_field_idx].squeeze(1)
    #     x = self.mlp(x.view(-1, self.embed_output_dim))
    #     gmf = user_x * item_x
    #     x = torch.cat([gmf, x], dim=1)
    #     x = self.fc(x).squeeze(1)
    #     return x