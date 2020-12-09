# Import NumPy and PyTorch and Pickle
import numpy as np
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score
import torch.utils.data
from torch.utils.data import DataLoader

# # Import PyTorch Ignite
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Loss
# from ignite.metrics import MeanSquaredError

# Import Tensorboard
# from tensorboardX import SummaryWriter

# Import Utility Functions
# from loader import Loader
from datetime import datetime

from RRN.data import RRNDataset
from RRN.model import *

# Import the NCF model
import NCF

n_item = 13522
n_tag = 188
q_padding_idx = n_item
tag_padding_idx = n_tag

# Load data for RRN

def get_dataset(name, path):
    """
    Get the dataset
    :param name: name of the dataset
    :param path: path to the dataset
    :return: RRNDataset
    """
    if name == 'RRNDataset':
        return RRNDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def pad_collate(batch):
  (users, questions, times, targets, tags) = zip(*batch)
  # print(len(questions))
  # print(questions[0])
  # TEMPORARILY CUTTING LENGTH IN DATASET
  # for i in range(len(questions)):
  #   if len(questions[i]) > 200:
  #       questions[i] = questions[i][:200]
  #       times[i] = times[i][:200]
  #       targets[i] = targets[i][:200]
  q_lens = [len(q) for q in questions]
  t_lens = [len(t) for t in targets]

  # print(q_lens)
  # print(t_lens)

  # print(np.shape(questions))
  # questions = torch.tensor(questions, dtype=torch.int32)
  # times = torch.tensor(times, dtype=torch.float)
  # targets = torch.tensor(np.asarray(targets, dtype=np.int8), dtype=torch.int32)

  questions_pad = pad_sequence(questions, batch_first=True, padding_value=q_padding_idx)
  times_pad = pad_sequence(times, batch_first=True, padding_value=0)
  targets_pad = pad_sequence(targets, batch_first=True, padding_value=0)
  tags_pad = pad_sequence(tags, batch_first=True, padding_value=tag_padding_idx)

  return users, questions_pad, times_pad, targets_pad, tags_pad, q_lens, t_lens


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    """
    Train the model
    :param model: choice of model
    :param optimizer: choice of optimizer
    :param data_loader: data loader class
    :param criterion: choice of loss function
    :param device: choice of device
    :return: loss being logged
    """
    # Step into train mode
    model.train()
    total_loss = 0

    max_seq_len = 200

    for k, (users, questions, times, targets, tags, q_lens, _) in enumerate(data_loader):
        questions, times, targets, tags = questions.to(device), times.to(device), targets.to(device), tags.to(device)

        batch_size = questions.shape[0]
        seq_len = questions.shape[1]

        y = model(questions, times, tags)

        ####### NEW METHOD FOR NCF INPUTS
        # ncf_inputs = torch.zeros(batch_size*seq_len, 2, dtype=torch.long).to(device)

        # for i in range(batch_size):
        #     start_idx = i*max_seq_len
        #     end_idx = start_idx + seq_len
        #     ncf_inputs[start_idx:end_idx, 0] = users[i]
        #     ncf_inputs[start_idx:end_idx, 1] = questions[i]

        # ncf_outputs = ncf(ncf_inputs).detach().reshape(batch_size,seq_len)
        
        # total_output = torch.clamp(y + ncf_outputs, 0, 1)
        ######
        total_output = y
        ######

        # IS THIS CORRECT????
        mask = questions != q_padding_idx

        total_output = torch.masked_select(total_output, mask)
        targets = torch.masked_select(targets.float().squeeze(), mask)

        loss = criterion(total_output, targets)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Log the total loss for every 1000 runs
        if (k + 1) % log_interval == 0:
            print('iteration: ', k, '    - loss:', total_loss / log_interval)
            total_loss = 0

        # wandb.log({"Total Loss": total_loss})

def test(model, data_loader, device):
    """
    Evaluate the model
    :param model: choice of model
    :param data_loader: data loader class
    :param device: choice of device
    :return: AUC score
    """
    # Step into evaluation mode
    model.eval()
    tgts, predicts, ncf_predicts = list(), list(), list()

    max_seq_len = 200

    with torch.no_grad():
        for k, (users, questions, times, targets, tags, q_lens, _) in enumerate(data_loader):

            questions, times, targets, tags = questions.to(device), times.to(device), targets.to(device), tags.to(device)

            batch_size = questions.shape[0]
            seq_len = questions.shape[1]

            y = model(questions, times, tags)

            # NEW METHOD FOR NCF INPUTS
            # ncf_inputs = torch.zeros(batch_size*seq_len, 2, dtype=torch.long).to(device)

            # for i in range(batch_size):
            #     start_idx = i*max_seq_len
            #     end_idx = start_idx + seq_len
            #     ncf_inputs[start_idx:end_idx, 0] = users[i]
            #     ncf_inputs[start_idx:end_idx, 1] = questions[i]

            # ncf_outputs = ncf(ncf_inputs).detach().reshape(batch_size,seq_len)

            # total_output = torch.clamp(y + ncf_outputs, 0, 1)

            ######
            total_output = y
            ######

            # IS THIS CORRECT????
            mask = questions != q_padding_idx
            outputs = torch.masked_select(total_output, mask)
            targets = torch.masked_select(targets.float().squeeze(), mask)

            # ncf_predicts.extend(torch.masked_select(ncf_outputs, mask).tolist())
            predicts.extend(outputs.tolist())
            tgts.extend(targets.tolist())

            if (k + 1) % 10 == 0:
                print('test iteration: ', k)


    # Return AUC score between predicted ratings and actual ratings
    # print('ncf roc:')
    # print(roc_auc_score(tgts, ncf_predicts))
    return roc_auc_score(tgts, predicts)


def main(dataset_name, dataset_path, model_name, epoch, learning_rate,
         batch_size, weight_decay, device, save_dir, pretrained):
    """
    Main function
    :param dataset_name: Choice of the dataset (MovieLens1M)
    :param dataset_path: Directory of the dataset
    :param model_name: Choice of the model
    :param epoch: Number of epochs
    :param learning_rate: Learning rate
    :param batch_size: Batch size
    :param weight_decay: Weight decay
    :param device: CHoice of device
    :param save_dir: Directory of the saved model
    :return: Saved model with logged AUC results
    """

    # Get the dataset
    dataset = get_dataset(dataset_name, dataset_path)
    # Split the data into 80% train, 10% validation, and 10% test
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length

    print("Training set length: ", train_length)
    print("Validation set length: ", valid_length)
    print("Test set length: ", test_length)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))

    # Instantiate data loader classes for train, validation, and test sets
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=pad_collate, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_collate, num_workers=8)

    # Get the model
    # model = get_model(model_name, dataset).to(device)
    model = UserTemp(embed_dim=10, hidden_dim=10, questionset_size=n_item, tagset_size=n_tag)
    model.to(device)

    if pretrained:
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(pretrained, map_location=device))


    # Use binary cross entropy loss
    criterion = torch.nn.BCELoss()
    # Use Adam optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Log metrics with Weights and Biases
    # wandb.watch(model, log="all")

    # Test the Test loop
    # valid_auc = test(model, valid_data_loader, device)
    # Log the epochs and AUC on the validation set
    # print('epoch: -1 validation: auc:', valid_auc)

    # Loop through pre-defined number of epochs
    for epoch_i in range(epoch):
        # Perform training on the train set
        train(model, optimizer, train_data_loader, criterion, device)
        torch.save(model.state_dict(), f'{save_dir}/{model_name}-{epoch_i}.pt')
        # Perform evaluation on the validation set
        valid_auc = test(model, valid_data_loader, device)
        # Log the epochs and AUC on the validation set
        print('epoch:', epoch_i, 'validation: auc:', valid_auc)
        # wandb.log({"Validation AUC": valid_auc})

    # Perform evaluation on the test set
    test_auc = test(model, test_data_loader, device)
    # Log the final AUC on the test set
    print('test auc:', test_auc)
    # wandb.log({"Test AUC": test_auc})

    # Save the model checkpoint
    torch.save(model.state_dict(), f'{save_dir}/{model_name}.pt')





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='RRNDataset')
    parser.add_argument('--dataset_path', default='../input/')
    parser.add_argument('--model_name', default='rrn')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='models')
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Import the user map and item map
    data_path = '../input/'
    model_path = 'models/'

    with open(data_path + 'user_map.pkl', 'rb') as f:
        user_map = pickle.load(f)
        
    # with open(data_path + 'item_map.pkl', 'rb') as f2:
    #     item_map = pickle.load(f2)

    # Load the NCF model
    n_user = len(user_map)
    print(n_user)

    field_dims = [n_user , n_item + 1]

    print(field_dims)

    ncf = NCF.NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.5,
                                                user_field_idx=0,
                                                item_field_idx=1)
    ncf.load_state_dict(torch.load(model_path + 'ncf.pt', map_location=device))
    ncf.to(device)
    ncf.eval()

    main(args.dataset_name, args.dataset_path, args.model_name, args.epoch, args.learning_rate,
         args.batch_size, args.weight_decay, device, args.save_dir, args.pretrained)



