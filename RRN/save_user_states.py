import torch
from RRN.data import RRNDataset
from RRNCF.model import RRNCF
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
    
n_item = 13523
n_tag = 188
q_padding_idx = n_item
tag_padding_idx = n_tag

def get_dataset(name, path):
    """
    Get the dataset
    :param name: name of the dataset
    :param path: path to the dataset
    :return: RRNDataset
    """
    print("Loading dataset...")

    if name == 'RRNDataset':
        return RRNDataset(path, truncate=False, short=True, return_users=True)
    else:
        raise ValueError('unknown dataset name: ' + name)

def pad_collate(batch):
  (users, questions, times, tags, targets) = zip(*batch)

  q_lens = [len(q) for q in questions]

  questions_pad = pad_sequence(questions, batch_first=True, padding_value=q_padding_idx)
  times_pad = pad_sequence(times, batch_first=True, padding_value=0)
  tags_pad = pad_sequence(tags, batch_first=True, padding_value=tag_padding_idx)
  targets_pad = pad_sequence(targets, batch_first=True, padding_value=0)

  return users, questions_pad, times_pad, tags_pad, targets_pad, q_lens


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = get_dataset('RRNDataset', '../input/')

user_states = torch.zeros((len(dataset), 16))

data_loader = DataLoader(dataset, batch_size=10, collate_fn=pad_collate, shuffle=False)

model = RRNCF(embed_dim=16, mlp_dim=16, dropout=0.2, questionset_size=n_item, tagset_size=n_tag)
model.to(device)

print("Loading pre-trained model...")
model.load_state_dict(torch.load('models/rrncf.pt', map_location=device))


model.eval()
with torch.no_grad():
    for k, (users, questions, times, tags, targets, q_lens) in enumerate(data_loader):

      print(users)
      batch_size = questions.shape[0]

      questions, times, tags, targets = questions.to(device), times.to(device), tags.to(device), targets.to(device)

      h_t = model.get_user_state(questions, times, tags, targets, q_lens)


      # print(y)
      # print(q_lens)
      
      # user_states[k*batch_size:(k+1)*batch_size] = h_t
      user_states[users] = h_t
      print(user_states[users[2]])

      if k%100 == 0:
        print("iteration: ",k)

torch.save(user_states, 'user_states.pt')
		
