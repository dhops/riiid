import numpy as np
import pandas as pd
import time
import sys
import pickle

out_path = '../input/'
in_path = '../input/data/'
TRUNCATE_LENGTH = 1000



# with open(out_path + 'q_tag_map.pkl', 'rb') as f:
#   q_tag_map = pickle.load(f)
# with open(out_path + 'l_tag_map.pkl', 'rb') as f:
#   l_tag_map = pickle.load(f)

# with open(out_path + 'datasetNCF.pkl', 'rb') as f: 
#   data = pickle.load(f)
# with open(out_path + 'user_map.pkl', 'rb') as f: 
#   user_map = pickle.load(f)
# # with open(path + 'item_map.pkl', 'rb') as f: 
# #   item_map = pickle.load(f)

with open(out_path + 'rrn_dict_items.pkl', 'rb') as f:
  rrn_dict_items = pickle.load(f)
with open(out_path + 'rrn_dict_times.pkl', 'rb') as f:
  rrn_dict_times = pickle.load(f)
with open(out_path + 'rrn_dict_ratings.pkl', 'rb') as f:
  rrn_dict_ratings = pickle.load(f)
with open(out_path + 'rrn_dict_tags.pkl', 'rb') as f:
  rrn_dict_tags = pickle.load(f)

for user in range(len(rrn_dict_items)):
  if len(rrn_dict_items[user]) > TRUNCATE_LENGTH:
    rrn_dict_items[user] = rrn_dict_items[user][-TRUNCATE_LENGTH:]
    rrn_dict_times[user] = rrn_dict_times[user][-TRUNCATE_LENGTH:]
    rrn_dict_tags[user] = rrn_dict_tags[user][-TRUNCATE_LENGTH:]
    rrn_dict_ratings[user] = rrn_dict_ratings[user][-TRUNCATE_LENGTH:]
  if user%1000 == 0:
    print("user #", user)


with open(out_path + 'items_short.pkl', 'wb') as f:
  pickle.dump(rrn_dict_items, f)
with open(out_path + 'times_short.pkl', 'wb') as f:
  pickle.dump(rrn_dict_times, f)
with open(out_path + 'ratings_short.pkl', 'wb') as f:
  pickle.dump(rrn_dict_ratings, f)
with open(out_path + 'tags_short.pkl', 'wb') as f:
  pickle.dump(rrn_dict_tags, f)