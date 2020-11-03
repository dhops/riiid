# Copyright (c) 2017 NVIDIA Corporation
"""Data Layer Classes"""
from os import listdir, path
from random import shuffle
import torch
import numpy as np
import pickle

class UserItemRecDataProvider:
  def __init__(self, params, user_id_map=None, item_id_map=None, test_set=False):
    self._params = params
    self._data_dir = self.params['data_dir']
    self._extension = ".txt" if 'extension' not in self.params else self.params['extension']
    self._i_id = 0 if 'itemIdInd' not in self.params else self.params['itemIdInd']
    self._u_id = 1 if 'userIdInd' not in self.params else self.params['userIdInd']
    self._r_id = 2 if 'ratingInd' not in self.params else self.params['ratingInd']
    self._major = 'items' if 'major' not in self.params else self.params['major']
    if not (self._major == 'items' or self._major == 'users'):
      raise ValueError("Major must be 'users' or 'items', but got {}".format(self._major))

    self._major_ind = self._i_id if self._major == 'items' else self._u_id
    self._minor_ind = self._u_id if self._major == 'items' else self._i_id
    self._delimiter = '\t' if 'delimiter' not in self.params else self.params['delimiter']

    # Hard-coded for Riiid Data

    data_full = np.load('../input/deepRecDataset.npz')

    if ~test_set:
      print('loading training data')
      if user_id_map is None or item_id_map is None:
        # self._build_maps()
        self._user_id_map = data_full['user_map'].item()
        self._item_id_map = data_full['item_map'].item()
      else:
        self._user_id_map = user_id_map
        self._item_id_map = item_id_map

      major_map = self._item_id_map if self._major == 'items' else self._user_id_map
      minor_map = self._user_id_map if self._major == 'items' else self._item_id_map

      self._vector_dim = len(minor_map)

      self._batch_size = self.params['batch_size']

      self.data = dict()

      try: 
        with open('data/train_set.pkl', 'rb') as f:
          self.data = pickle.load(f)
          print(type(self.data[0]))
      except:
        train_set = data_full['train_set']

        for i in range(len(train_set)):
          key = major_map[train_set[i][self._major_ind]]
          value = minor_map[train_set[i][self._minor_ind]]
          rating = float(train_set[i][self._r_id])
          # print("Key: {}, Value: {}, Rating: {}".format(key, value, rating))
          if key not in self.data:
            self.data[key] = []
          self.data[key].append((value, rating))
          if i%1000000 == 0:
            print(i)
        with open('data/train_set.pkl', 'wb') as f:
          pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
    else:
      print('loading testing data')
      if user_id_map is None or item_id_map is None:
        # self._build_maps()
        self._user_id_map = data_full['user_map'].item()
        self._item_id_map = data_full['item_map'].item()
      else:
        self._user_id_map = user_id_map
        self._item_id_map = item_id_map

      major_map = self._item_id_map if self._major == 'items' else self._user_id_map
      minor_map = self._user_id_map if self._major == 'items' else self._item_id_map

      self._vector_dim = len(minor_map)

      self._batch_size = self.params['batch_size']

      self.data = dict()

      try: 
        with open('data/test_set.pkl', 'rb') as f:
          self.data = pickle.load(f)
      except:
        test_set = data_full['test_set']
        print('here')
        for i in range(len(test_set)):
          key = major_map[test_set[i][self._major_ind]]
          value = minor_map[test_set[i][self._minor_ind]]
          rating = float(test_set[i][self._r_id])
          # print("Key: {}, Value: {}, Rating: {}".format(key, value, rating))
          if key not in self.data:
            self.data[key] = []
          self.data[key].append((value, rating))
          if i%1000 == 0:
            print(i)
        with open('data/test_set.pkl', 'wb') as f:
          pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)


    # if user_id_map is None or item_id_map is None:
    #   self._build_maps()
    # else:
    #   self._user_id_map = user_id_map
    #   self._item_id_map = item_id_map

    # major_map = self._item_id_map if self._major == 'items' else self._user_id_map
    # minor_map = self._user_id_map if self._major == 'items' else self._item_id_map
    # self._vector_dim = len(minor_map)

    # src_files = [path.join(self._data_dir, f)
    #               for f in listdir(self._data_dir)
    #               if path.isfile(path.join(self._data_dir, f)) and f.endswith(self._extension)]

    # self._batch_size = self.params['batch_size']

    # self.data = dict()

    # for source_file in src_files:
    #   with open(source_file, 'r') as src:
    #     for line in src.readlines():
    #       parts = line.strip().split(self._delimiter)
    #       if len(parts)<3:
    #         raise ValueError('Encountered badly formatted line in {}'.format(source_file))
    #       key = major_map[int(parts[self._major_ind])]
    #       value = minor_map[int(parts[self._minor_ind])]
    #       rating = float(parts[self._r_id])
    #       #print("Key: {}, Value: {}, Rating: {}".format(key, value, rating))
    #       if key not in self.data:
    #         self.data[key] = []
    #       self.data[key].append((value, rating))



  def _build_maps(self):
    self._user_id_map = dict()
    self._item_id_map = dict()

    src_files = [path.join(self._data_dir, f)
                 for f in listdir(self._data_dir)
                 if path.isfile(path.join(self._data_dir, f)) and f.endswith(self._extension)]

    u_id = 0
    i_id = 0
    for source_file in src_files:
      with open(source_file, 'r') as src:
        for line in src.readlines():
          parts = line.strip().split(self._delimiter)
          if len(parts)<3:
            raise ValueError('Encountered badly formatted line in {}'.format(source_file))

          u_id_orig = int(parts[self._u_id])
          if u_id_orig not in self._user_id_map:
            self._user_id_map[u_id_orig] = u_id
            u_id += 1

          i_id_orig = int(parts[self._i_id])
          if i_id_orig not in self._item_id_map:
            self._item_id_map[i_id_orig] = i_id
            i_id += 1


  def iterate_one_epoch(self):
    data = self.data
    keys = list(data.keys())
    shuffle(keys)
    s_ind = 0
    e_ind = self._batch_size
    while e_ind < len(keys):
      local_ind = 0
      inds1 = []
      inds2 = []
      vals = []
      for ind in range(s_ind, e_ind):
        inds2 += [v[0] for v in data[keys[ind]]]
        inds1 += [local_ind]*len([v[0] for v in data[keys[ind]]])
        vals += [v[1] for v in data[keys[ind]]]
        local_ind += 1

      i_torch = torch.LongTensor([inds1, inds2])
      v_torch = torch.FloatTensor(vals)

      mini_batch = torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([self._batch_size, self._vector_dim]))
      s_ind += self._batch_size
      e_ind += self._batch_size
      yield  mini_batch

  def iterate_one_epoch_eval(self, for_inf=False):
    keys = list(self.data.keys())
    s_ind = 0
    while s_ind < len(keys):
      inds1 = [0] * len([v[0] for v in self.data[keys[s_ind]]])
      inds2 = [v[0] for v in self.data[keys[s_ind]]]
      vals = [v[1] for v in self.data[keys[s_ind]]]

      src_inds1 = [0] * len([v[0] for v in self.src_data[keys[s_ind]]])
      src_inds2 = [v[0] for v in self.src_data[keys[s_ind]]]
      src_vals = [v[1] for v in self.src_data[keys[s_ind]]]

      i_torch = torch.LongTensor([inds1, inds2])
      v_torch = torch.FloatTensor(vals)

      src_i_torch = torch.LongTensor([src_inds1, src_inds2])
      src_v_torch = torch.FloatTensor(src_vals)

      mini_batch = (torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([1, self._vector_dim])),
                    torch.sparse.FloatTensor(src_i_torch, src_v_torch, torch.Size([1, self._vector_dim])))
      s_ind += 1
      if not for_inf:
        yield  mini_batch
      else:
        yield mini_batch, keys[s_ind - 1]

  @property
  def vector_dim(self):
    return self._vector_dim

  @property
  def userIdMap(self):
    return self._user_id_map

  @property
  def itemIdMap(self):
    return self._item_id_map

  @property
  def params(self):
    return self._params
