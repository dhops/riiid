import numpy as np
import pandas as pd
import time
import sys
import pickle

out_path = '../input/'
in_path = '../input/data/'


try: 
  with open(out_path + 'q_tag_map.pkl', 'rb') as f:
    q_tag_map = pickle.load(f)
  with open(out_path + 'l_tag_map.pkl', 'rb') as f:
    l_tag_map = pickle.load(f)
except Exception as e:
  print("Generating tags maps")

  q_data = pd.read_csv(in_path + 'questions.csv')
  l_data = pd.read_csv(in_path + 'lectures.csv')

  q_data = q_data[['question_id','tags']]
  l_data = l_data[['lecture_id','tag']]

  print(l_data.head())

  q_data['tags'] = q_data['tags'].str.split()
  def tag_to_int(t_list):
    return [int(i) for i in t_list]
  q_data['tags'] = q_data['tags'].map(tag_to_int, na_action='ignore')

  q_tag_map = {}
  for i in range(len(q_data)):
    q_tag_map[q_data.loc[i]['question_id']] = q_data.loc[i]['tags']

  l_tag_map = {}
  for i in range(len(l_data)):
    l_tag_map[l_data.loc[i]['lecture_id']] = [l_data.loc[i]['tag']]

  with open(out_path + 'q_tag_map.pkl', 'wb') as f:
    pickle.dump(q_tag_map, f)
  with open(out_path + 'l_tag_map.pkl', 'wb') as f:
    pickle.dump(l_tag_map, f)






try:
  with open(out_path + 'datasetNCF.pkl', 'rb') as f: 
    data = pickle.load(f)
  with open(out_path + 'user_map.pkl', 'rb') as f: 
    user_map = pickle.load(f)
  # with open(path + 'item_map.pkl', 'rb') as f: 
  #   item_map = pickle.load(f)
except Exception as e:
  print("Preparing NCF Dataset...")

  data = pd.read_feather(in_path + "train.feather")

  data.rename(columns={"user_id": "user", "content_id": "item", "answered_correctly": "rating"}, inplace=True)
  # data = data[data.rating != -1]
  data = data[['user','item','rating','timestamp']]
  # print(data[data.rating == -1])
  # data.loc[data.rating == -1, 'item'] += 20000
  # data.loc[data.rating == -1, 'rating'] = 1

  print(data[data.rating == -1])

  def id_to_tags(r, id):
    if r == -1:
      return l_tag_map[id]
    else:
      return q_tag_map[id]

  # print(data[80:100,:])

  # data['tags'] = data.apply(lambda row: id_to_tags(r = row['rating'], id = row['item']), axis=1)

  start_time = time.time()
  print("--- %s seconds ---" % (time.time() - start_time))
  data['tags'] = [id_to_tags(*a) for a in tuple(zip(data['rating'], data['item']))]

  print("--- %s seconds ---" % (time.time() - start_time))

  print(data.head())
  # print(data[80:100,:])

  data.loc[data.rating == -1, 'item'] = 13523
  print(data[data.rating == -1])
  data.loc[data.rating == -1, 'rating'] = 1
  print(data[data.rating == -1])

  # data = data.astype({'user': 'int32','item': 'int32','rating': 'bool','timestamp':'float'})

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

  # print("question padding: ", data['item'].max())

  data['user'] = data['user'].apply(lambda u: user_map[u])
  # data['item'] = data['item'].apply(lambda i: item_map[i])

  print(data.head(100))

  n_user = data.user.nunique()
  n_item = data.item.nunique()

  print("num users: ", n_user)
  print("num questions (including padding): ", n_item)


  with open(out_path + 'datasetNCF.pkl', 'wb') as f:
    pickle.dump(data, f)
  with open(out_path + 'user_map.pkl', 'wb') as f:
    pickle.dump(user_map, f)
  # with open(path + 'item_map.pkl', 'wb') as f:
  #   pickle.dump(item_map, f)

try: 
  with open(out_path + 'rrn_dict_items.pkl', 'rb') as f:
    rrn_dict_items = pickle.load(f)
  with open(out_path + 'rrn_dict_times.pkl', 'rb') as f:
    rrn_dict_times = pickle.load(f)
  with open(out_path + 'rrn_dict_ratings.pkl', 'rb') as f:
    rrn_dict_ratings = pickle.load(f)
  with open(out_path + 'rrn_dict_tags.pkl', 'rb') as f:
    rrn_dict_tags = pickle.load(f)
except Exception as e:
  print("Preparing RRN dicts...")

  # MAKE GAPS
  print(data['timestamp'].head(100))
  data['timestamp'] = data['timestamp'] - data['timestamp'].shift(1)
  print(data['timestamp'].head(100))
  data.loc[data.groupby('user')['timestamp'].head(1).index, 'timestamp'] = 0
  print(data['timestamp'].head(100))

  #0-1 scale for gaps
  # data['timestamp'] = (data['timestamp'] - data['timestamp'].min()) / (data['timestamp'].max() - data['timestamp'].min())
  # print(data['timestamp'].head(100))

  data['timestamp'] = (data['timestamp']-data['timestamp'].mean()) / data['timestamp'].std()
  print(data['timestamp'].head(100))

  rrn_dict_tags = data.groupby('user')['tags'].apply(list).to_dict()
  rrn_dict_items = data.groupby('user')['item'].apply(list).to_dict()
  rrn_dict_times = data.groupby('user')['timestamp'].apply(list).to_dict()
  rrn_dict_ratings = data.groupby('user')['rating'].apply(list).to_dict()

  # rrn_dict = {k: list(v) for k, v in data.groupby('user')[['item','timestamp']]}

  # print(rrn_dict_items[0])
  print(rrn_dict_tags[0])
  print(rrn_dict_tags[1])
  print(rrn_dict_times[0])
  print(rrn_dict_times[1])
  # print(rrn_dict_ratings[0])

  with open(out_path + 'rrn_dict_items.pkl', 'wb') as f:
    pickle.dump(rrn_dict_items, f)
  with open(out_path + 'rrn_dict_times.pkl', 'wb') as f:
    pickle.dump(rrn_dict_times, f)
  with open(out_path + 'rrn_dict_ratings.pkl', 'wb') as f:
    pickle.dump(rrn_dict_ratings, f)
  with open(out_path + 'rrn_dict_tags.pkl', 'wb') as f:
    pickle.dump(rrn_dict_tags, f)


# try: 
#   with open(out_path + 'tag_map.pkl', 'rb') as f:
#     tag_map = pickle.load(f)
# except Exception as e:
#   print("Generating tags dict")
#   print(e)
#   q_data = pd.read_csv(in_path + 'questions.csv')
#   q_data = q_data[['question_id','tags']]
#   print(q_data.head(5))
#   # q_data['question_id'] = q_data['question_id'].apply(lambda i: item_map[i])
#   q_data['tags'] = q_data['tags'].str.split()

#   def tag_to_int(t_list):
#     return [int(i) for i in t_list]

#   q_data['tags'] = q_data['tags'].map(tag_to_int, na_action='ignore')

#   l_data = pd.read_csv(in_path + 'lectures.csv')


#   tag_map = {}
#   for i in range(len(q_data)):
#     tag_map[q_data.loc[i]['question_id']] = q_data.loc[i]['tags']

#   with open(out_path + 'tag_map.pkl', 'wb') as f:
#     pickle.dump(tag_map, f)


# try: 
#   with open(out_path + 'rrn_dict_tags.pkl', 'rb') as f:
#     rrn_dict_tags = pickle.load(f)
# except Exception as e:
#   print("Generating tags dict")
#   print(e)
#   q_data = pd.read_csv(in_path + 'questions.csv')
#   q_data = q_data[['question_id','tags']]
#   print(q_data.head(5))
#   # q_data['question_id'] = q_data['question_id'].apply(lambda i: item_map[i])
#   q_data['tags'] = q_data['tags'].str.split()

#   def tag_to_int(t_list):
#     return [int(i) for i in t_list]

#   q_data['tags'] = q_data['tags'].map(tag_to_int, na_action='ignore')

#   rrn_dict_tags = {}
#   for u in range(len(user_map)):
#     for q in range(len(rrn_dict_items[u])):
#       if 
#       rrn_dict_tags[u]




#   l_data = pd.read_csv(in_path + 'lectures.csv')


#   tag_map = {}
#   for i in range(len(q_data)):
#     tag_map[q_data.loc[i]['question_id']] = q_data.loc[i]['tags']

#   with open(out_path + 'rrn_dict_tags.pkl', 'wb') as f:
#     pickle.dump(tag_map, f)



