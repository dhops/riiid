# Import NumPy and PyTorch and Pickle
import numpy as np
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score
import torch.utils.data
from torch.utils.data import DataLoader, Subset

# # Import PyTorch Ignite
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Loss
# from ignite.metrics import MeanSquaredError

# Import Tensorboard
# from tensorboardX import SummaryWriter

# Import Utility Functions
# from loader import Loader

from RRN.data import RRNDataset
from RRNCF.model import RRNCF

n_item = 13523
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
    print("Loading dataset...")

    if name == 'RRNDataset':
        return RRNDataset(path, truncate=True, short=True)
    else:
        raise ValueError('unknown dataset name: ' + name)


def pad_collate(batch):
  (questions, times, tags, targets) = zip(*batch)

  q_lens = [len(q) for q in questions]

  questions_pad = pad_sequence(questions, batch_first=True, padding_value=q_padding_idx)
  times_pad = pad_sequence(times, batch_first=True, padding_value=0)
  tags_pad = pad_sequence(tags, batch_first=True, padding_value=tag_padding_idx)
  targets_pad = pad_sequence(targets, batch_first=True, padding_value=0)

  return questions_pad, times_pad, tags_pad, targets_pad, q_lens


def train(model, optimizer, data_loader, criterion, device, log_interval=100, base=None):
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
    log_steps = 0

    for k, (questions, times, tags, targets, q_lens) in enumerate(data_loader):

            questions, times, tags, targets = questions.to(device), times.to(device), tags.to(device), targets.to(device)

            y = model.forward(questions, times, tags, targets, q_lens)

            y = y.reshape(questions.shape)

            mask = questions != q_padding_idx
            y = torch.masked_select(y, mask)
            targets = torch.masked_select(targets.float().reshape(mask.shape), mask)

            loss = criterion(y, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            log_steps += 1
        # Log the total loss for every log_interval runs
            if (log_steps + 1) % log_interval == 0:
                print('iteration: ', log_steps, '    - loss:', total_loss / log_interval)
                total_loss = 0

        # wandb.log({"Total Loss": total_loss})

def test2(model, data_loader, device, base):
    """
    Evaluate the model
    :param model: choice of model
    :param data_loader: data loader class
    :param device: choice of device
    :return: AUC score
    """
    # Step into evaluation mode
    model.eval()
    tgts, predicts = list(), list()
    tgts_last, predicts_last = list(), list()

    with torch.no_grad():
        for k, (questions, times, tags, targets, q_lens) in enumerate(data_loader):
            
            questions, times, tags, targets = questions.to(device), times.to(device), tags.to(device), targets.to(device)

            y = model(questions, times, tags, targets, q_lens)

            y = y.reshape(questions.shape)

            mask = questions != q_padding_idx

            q_idx = [l-1 for l in q_lens]
            y_last = y[np.arange(y.shape[0]), q_idx].flatten()
            t_last = targets[np.arange(y.shape[0]), q_idx].flatten()

            y = torch.masked_select(y, mask)
            targets = torch.masked_select(targets.float().reshape(mask.shape), mask)

            predicts.extend(y.tolist())
            tgts.extend(targets.tolist())

            predicts_last.extend(y_last.tolist())
            tgts_last.extend(t_last.tolist())

            if (k + 1) % 10 == 0:
                print('test iteration: ', k)

    print("roc auc last only: ", roc_auc_score(tgts_last, predicts_last))

    # Return AUC score between predicted ratings and actual ratings
    return roc_auc_score(tgts, predicts)


def main(dataset_name, dataset_path, model_name, epoch, learning_rate,
         batch_size, weight_decay, device, save_dir, pretrained, base, log_interval):
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
    :param pretrained: pre-trained model to continue training
    :param base: base model type (None, MF, NCF, etc.)
    :return: Saved model with logged AUC results
    """

    # Get the dataset
    dataset = get_dataset(dataset_name, dataset_path)
    # Split the data into 80% train, 10% validation, and 10% test
    # train_length = int(len(dataset) * 0.8)
    # valid_length = int(len(dataset) * 0.1)
    # test_length = len(dataset) - train_length - valid_length

    # print("Training set length: ", train_length)
    # print("Validation set length: ", valid_length)
    # print("Test set length: ", test_length)

    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, (train_length, valid_length, test_length))
    print("Dataset length: ", len(dataset))

    # train_indices = np.sort(np.random.permutation(len(dataset))[:10000]).tolist()
    train_indices = list(range(len(dataset)))
    valid_indices = np.sort(np.random.permutation(len(dataset))[:10000]).tolist()
    test_indices = np.sort(np.random.permutation(len(dataset))[:10000]).tolist()

    for i in valid_indices:
        train_indices.remove(i)

    # train_dataset = Subset(dataset, train_indices)
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    print(len(train_dataset))
    print(len(valid_dataset))

    # Instantiate data loader classes for train, validation, and test sets
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=4, drop_last=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=1, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=1, drop_last=True)

    # Get the model
    # model = get_model(model_name, dataset).to(device)
    model = RRNCF(embed_dim=16, mlp_dim=16, dropout=0.2, questionset_size=n_item, tagset_size=n_tag)
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

    # Test the pre-trained model
    # valid_auc = test2(model, valid_data_loader, device, base)
    # print('epoch: -1 validation: auc:', valid_auc)

    # Loop through pre-defined number of epochs
    for epoch_i in range(epoch):
        # Perform training on the train set
        train(model, optimizer, train_data_loader, criterion, device, log_interval, base)
        torch.save(model.state_dict(), f'{save_dir}/{model_name}-{epoch_i}.pt')
        # Perform evaluation on the validation set
        valid_auc = test2(model, valid_data_loader, device, base)
        # Log the epochs and AUC on the validation set
        print('epoch:', epoch_i + 1, 'validation: auc:', valid_auc)
        # wandb.log({"Validation AUC": valid_auc})

    # Perform evaluation on the test set
    test_auc = test2(model, test_data_loader, device, base)
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
    parser.add_argument('--model_path', default='models/')
    parser.add_argument('--model_name', default='rrn')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='models')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the base model
    main(args.dataset_name, args.dataset_path, args.model_name, args.epoch, args.learning_rate,
         args.batch_size, args.weight_decay, device, args.save_dir, args.pretrained, args.base, args.log_interval)



