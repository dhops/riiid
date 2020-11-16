import numpy as np
import pandas as pd
import time
import sys
import pickle

path = '../input/'

try: 
  data = pickle.load(open(path + "datasetNCF.pkl"))
  user_map = pickle.load(open(path + "user_map.pkl"))
  item_map = pickle.load(open(path + "item_map.pkl"))
except:
  data = pd.read_feather(path + "riiid_train.feather")

  # print(data.head(100))

  data.rename(columns={"user_id": "user", "content_id": "item", "answered_correctly": "rating"}, inplace=True)
  data = data[data.rating != -1]
  data = data[['user','item','rating','timestamp']]

  timestamp_mean = data['timestamp'].mean()
  timestamp_std = data['timestamp'].std()

  data['timestamp'] = (data['timestamp'] - timestamp_mean)/timestamp_std

  data = data.astype({'user': 'int32','item': 'int32','rating': 'bool','timestamp':'float'})

  print(data.head(100))
  print(type(data['user']))

  # Make User Map and Item Map

  old_user_ids = data['user']
  new_user_ids, ___ = data['user'].factorize()
  user_map = dict(set(zip(list(old_user_ids),list(new_user_ids))))

  old_item_ids = data['item']
  new_item_ids, ___ = data['item'].factorize()
  item_map = dict(set(zip(list(old_item_ids),list(new_item_ids))))

  # Map users and items according to maps

  data['user'] = data['user'].apply(lambda u: user_map[u])
  data['item'] = data['item'].apply(lambda i: item_map[i])

  print(data.head(100))
  print(type(data['user']))

  n_user = data.user.nunique()
  n_item = data.item.nunique()

  print(n_user)
  print(n_item)
  print(len(user_map))
  print(len(item_map))

  data.to_pickle(path + 'datasetNCF.pkl')
  with open(path + 'user_map.pkl') as f:
    pickle.dump(user_map, f)
  with open(path + 'item_map.pkl') as f:
    pickle.dump(item_map, f)


