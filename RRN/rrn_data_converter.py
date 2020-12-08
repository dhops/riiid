import numpy as np
import pandas as pd
import time
import sys
import pickle

path = '../input/'

try:
  with open(path + 'datasetNCF.pkl', 'rb') as f: 
    data = pickle.load(f)
  with open(path + 'user_map.pkl', 'rb') as f: 
    user_map = pickle.load(f)
  # with open(path + 'item_map.pkl', 'rb') as f: 
  #   item_map = pickle.load(f)
except Exception as e:
  print(e)

  print("Preparing NCF Dataset")

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

  # Don't use index 0, reserve for RNN padding
  # new_user_ids += 1

  user_map = dict(set(zip(list(old_user_ids),list(new_user_ids))))

  # old_item_ids = data['item']
  # new_item_ids, ___ = data['item'].factorize()

  # Don't use index 0, reserve for RNN padding
  # new_item_ids += 1

  # item_map = dict(set(zip(list(old_item_ids),list(new_item_ids))))

  # Map users and items according to maps

  print("question padding: ", data['item'].max()+1)

  data['user'] = data['user'].apply(lambda u: user_map[u])
  # data['item'] = data['item'].apply(lambda i: item_map[i])

  print(data.head(100))
  print(type(data['user']))

  n_user = data.user.nunique()
  n_item = data.item.nunique()

  print("num users: ", n_user)
  print("question padding (same?): ", n_item)
  # print(len(user_map))
  # print(len(item_map))

  data.to_pickle(path + 'datasetNCF.pkl')
  with open(path + 'user_map.pkl', 'wb') as f:
    pickle.dump(user_map, f)
  # with open(path + 'item_map.pkl', 'wb') as f:
  #   pickle.dump(item_map, f)

try: 
  with open(path + 'rrn_dict_items.pkl', 'rb') as f:
    rrn_dict_items = pickle.load(f)
  with open(path + 'rrn_dict_times.pkl', 'rb') as f:
    rrn_dict_times = pickle.load(f)
  with open(path + 'rrn_dict_ratings.pkl', 'rb') as f:
    rrn_dict_ratings = pickle.load(f)
except Exception as e:
  print(e)

  rrn_dict_items = data.groupby('user')['item'].apply(list).to_dict()
  rrn_dict_times = data.groupby('user')['timestamp'].apply(list).to_dict()
  rrn_dict_ratings = data.groupby('user')['rating'].apply(list).to_dict()

  # rrn_dict = {k: list(v) for k, v in data.groupby('user')[['item','timestamp']]}

  print(rrn_dict_items[0])
  print(rrn_dict_times[0])
  print(rrn_dict_ratings[0])

  with open(path + 'rrn_dict_items.pkl', 'wb') as f:
    pickle.dump(rrn_dict_items, f)
  with open(path + 'rrn_dict_times.pkl', 'wb') as f:
    pickle.dump(rrn_dict_times, f)
  with open(path + 'rrn_dict_ratings.pkl', 'wb') as f:
    pickle.dump(rrn_dict_ratings, f)


try: 
  with open(path + 'rrn_dict_tags.pkl', 'rb') as f:
    tag_map = pickle.load(f)
except Exception as e:
  print("Generating tags dict")
  print(e)
  q_data = pd.read_csv(path + 'questions.csv')
  q_data = q_data[['question_id','tags']]
  print(q_data.head(5))
  # q_data['question_id'] = q_data['question_id'].apply(lambda i: item_map[i])
  q_data['tags'] = q_data['tags'].str.split()

  def tag_to_int(t_list):
    return [int(i) for i in t_list]

  q_data['tags'] = q_data['tags'].map(tag_to_int, na_action='ignore')

  tag_map = {}
  for i in range(len(q_data)):
    tag_map[q_data.loc[i]['question_id']] = q_data.loc[i]['tags']

  with open(path + 'rrn_dict_tags.pkl', 'wb') as f:
    pickle.dump(tag_map, f)



