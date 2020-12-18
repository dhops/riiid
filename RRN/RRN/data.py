import numpy as np
import pandas as pd
import pickle
import torch
import os
from statistics import mean

class RRNDataset(torch.utils.data.Dataset):
    """
        RRN Riiid Full Dataset
    """

    def __init__(self, dataset_path, sep=',', engine='python', header=None):
        # Read the data into a Pandas dataframe
        # data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        # full_data = np.load(dataset_path)
        # Retrieve the items and ratings data
        with open(dataset_path + 'rrn_dict_items.pkl', 'rb') as f:
            self.rrn_dict_items = pickle.load(f)
        with open(dataset_path + 'rrn_dict_times.pkl', 'rb') as f:
            self.rrn_dict_times = pickle.load(f)
        with open(dataset_path + 'rrn_dict_ratings.pkl', 'rb') as f:
            self.rrn_dict_ratings = pickle.load(f)
        with open(dataset_path + 'rrn_dict_tags.pkl', 'rb') as f:
            self.rrn_dict_tags = pickle.load(f)
        # with open(dataset_path + 'item_map.pkl', 'rb') as f:
        #     self.item_map = pickle.load(f)

        self.tagset_size = 188
        self.tag_padding_idx = self.tagset_size
        self.max_tags = 6
        self.max_seq_len = 200

        # print("Processing Tags...")
        # self.tags = torch.ones((len(self.rrn_dict_tags), max_tags), dtype=torch.long) * tag_padding_idx
        # for k, v in self.rrn_dict_tags.items():
        #     if type(v) is list:
        #         for i in range(len(v)):
        #             self.tags[k, i] = v[i]



        # CHANGE T/F to 1/0 if needed.

        # print(self.rrn_dict_ratings[0])
        # for u in range(len(self.rrn_dict_ratings)):
        #     self.rrn_dict_ratings[u] = [int(r) for r in self.rrn_dict_ratings[u]]

        # print(self.rrn_dict_ratings[0])


        # self.items = full_data[['user','item']].values.astype(np.int)
        # self.targets = full_data['rating'].values.astype(np.float32)

        # Get the range of the items
        # self.field_dims = np.max(self.items, axis=0) + 1

        # Initialize NumPy arrays to store user and item indices
        # self.user_field_idx = np.array((0,), dtype=np.long)
        # self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        """
        :return: number of total users in dict
        """
        return len(self.rrn_dict_ratings)

    def __getitem__(self, index):
        """
        :param index: current index
        :return: the items, timestamps and ratings at current index
        """
        user = torch.tensor(index)
        questions = torch.from_numpy(np.asarray(self.rrn_dict_items[index]))
        times = torch.from_numpy(np.asarray(self.rrn_dict_times[index]))
        targets = torch.from_numpy(np.asarray(self.rrn_dict_ratings[index]))

        # tags = torch.ones((len(questions), max_tags), dtype=torch.long) * tag_padding_idx
        tags=np.array([tag+[self.tag_padding_idx]*(self.max_tags-len(tag)) for tag in self.rrn_dict_tags[index]])
        tags = torch.from_numpy(tags)

        if len(questions) > self.max_seq_len:
            questions = questions[:self.max_seq_len:]
            times = times[:self.max_seq_len:]
            targets = targets[:self.max_seq_len:]
            tags = tags[:self.max_seq_len:,:]
        return user, questions, times, tags, targets
        # return self.rrn_dict_items[index], self.rrn_dict_times[index], self.rrn_dict_ratings[index]

