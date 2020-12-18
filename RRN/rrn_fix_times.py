import numpy as np
import pandas as pd
import time
import sys
import pickle

path = '../input/'



# with open(path + 'user_map.pkl', 'rb') as f: 
# 	user_map = pickle.load(f)

# data = pd.read_feather(path + "riiid_train.feather")

# data.rename(columns={"user_id": "user", "content_id": "item", "answered_correctly": "rating"}, inplace=True)
# data = data[data.rating != -1]
# data = data[['user','item','rating','timestamp']]

# print(user_map[data['user'][0]])
# print(data.head())

# data['user'] = data['user'].apply(lambda u: user_map[u])

# print(data.head())

# rrn_dict_times = data.groupby('user')['timestamp'].apply(list).to_dict()

# print(rrn_dict_times[1])

# with open(path + 'rrn_dict_times.pkl', 'wb') as f:
#     pickle.dump(rrn_dict_times, f)



