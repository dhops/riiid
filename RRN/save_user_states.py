import torch
from RRN.data import RRNDataset
from RRNCF.model import RRNCF
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
    
n_item = 13522
n_tag = 188
q_padding_idx = n_item
tag_padding_idx = n_tag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RRNCF(embed_dim=16, mlp_dim=16, dropout=0.5, questionset_size=n_item, tagset_size=n_tag)
model.to(device)

print("Loading pre-trained model...")
model.load_state_dict(torch.load('models/rrn.pt', map_location=device))

def get_dataset(name, path):
    """
    Get the dataset
    :param name: name of the dataset
    :param path: path to the dataset
    :return: RRNDataset
    """
    print("Loading dataset...")

    if name == 'RRNDataset':
        return RRNDataset(path)
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


dataset = get_dataset('RRNDataset', '../input/')

user_states = np.zeros((len(dataset), 16))

data_loader = DataLoader(dataset, batch_size=10, collate_fn=pad_collate, shuffle=False)

model.eval()
with torch.no_grad():
    for k, (users, questions, times, tags, targets, q_lens) in enumerate(data_loader):
        
        questions, times, tags, targets = questions.to(device), times.to(device), tags.to(device), targets.to(device)

        y = model.get_user_state(questions, times, tags, targets)

        mask = questions != q_padding_idx

        # print(y)
        # print(q_lens)
        
        for i in range(len(users)):
        	user_states[users[i]] = y[i,q_lens[i]-1,:]
        	# print(y[i,q_lens[i]-1,:])
        	# print(users[i])

       	if k%100 == 0:
       		print("iteration: ", k)
			
with open('../user_states.pkl', 'wb') as f:
    pickle.dump(user_states, f)
print("done")

