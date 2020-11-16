# Import NumPy and PyTorch and Pickle
import numpy as np
import torch
import pickle


# Import PyTorch Ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.metrics import MeanSquaredError

# Import Tensorboard
from tensorboardX import SummaryWriter

# Import Utility Functions
from loader import Loader
from datetime import datetime

# Import the Model Script
from MF import *

# Import the NCF model
from NCF import *

# Import the user map and item map
data_path = '../input/'
model_path = '../models/'

with open(data_path + 'user_map.pkl', 'rb') as f1:
    user_map = pickle.load(f1)
    
with open(data_path + 'item_map.pkl', 'rb') as f2:
    rank_map = pickle.load(f2)

# Load the NCF model
n_user = len(user_map)
n_item = len(item_map)
field_dims = [n_user , n_item]

model = NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.5,
                                            user_field_idx=0,
                                            item_field_idx=1)
model.load_state_dict(torch.load(model_path + 'ncf.pt', map_location=torch.device('cpu')))
model.eval()



# Load data for RRN













# Define the Hyper-parameters
lr = 1e-2  # Learning Rate
k = 10  # Number of dimensions per user, item
c_vector = 1e-6  # regularization constant

# Setup TensorBoard logging
log_dir = 'runs/simple_mf_01_' + str(datetime.now()).replace(' ', '_')
writer = SummaryWriter(log_dir=log_dir)

# Instantiate the MF class object
model = MF(n_user, n_item, writer=writer, k=k, c_vector=c_vector)

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create a supervised trainer
trainer = create_supervised_trainer(model, optimizer, model.loss)

# Use Mean Squared Error as evaluation metric
metrics = {'evaluation': MeanSquaredError()}

# Create a supervised evaluator
evaluator = create_supervised_evaluator(model, metrics=metrics)

# Load the train and test data
train_loader = Loader(train_x, train_y, batchsize=1024)
test_loader = Loader(test_x, test_y, batchsize=1024)


def log_training_loss(engine, log_interval=500):
    """
    Function to log the training loss
    """
    model.itr = engine.state.iteration  # Keep track of iterations
    if model.itr % log_interval == 0:
        fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
        # Keep track of epochs and outputs
        msg = fmt.format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
        print(msg)


trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)


def log_validation_results(engine):
    """
    Function to log the validation loss
    """
    # When triggered, run the validation set
    evaluator.run(test_loader)
    # Keep track of the evaluation metrics
    avg_loss = evaluator.state.metrics['evaluation']
    print("Epoch[{}] Validation MSE: {:.2f} ".format(engine.state.epoch, avg_loss))
    writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)


trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

# Run the model for 50 epochs
trainer.run(train_loader, max_epochs=1)

# Save the model to a separate folder
torch.save(model.state_dict(), '../models/vanilla_mf.pth')
