import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 20,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 393656,
              'num_items': 13782,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': False,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 200,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 7,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 7,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }


# 'row_id': 'int64',
#     'timestamp': 'int64',
#     'user_id': 'int32',
#     'content_id': 'int16',
#     'content_type_id': 'int8',
# #     'task_container_id': 'int16',
# #     'user_answer': 'int8',
#     'answered_correctly': 'int8',
#     'prior_question_elapsed_time': 'float16',
#     'prior_question_had_explanation': 'boolean'

# Load Data
# riiid_dir = '../../input/riiid-test-answer-prediction/train.csv'
# # Rating refers to boolean "Answered correctly"
# riiid_data = pd.read_csv(riiid_dir, sep=',', 
#   # nrows=10**6, 
#   header=None, skiprows=1, names=['row_id', 'timestamp', 'user_id', 'content_id', 'content_type_id', 'task_container_id', 'user_answer', 'rating', 'prior_question_elapsed_time', 'prior_question_had_explanation'],  engine='python')
# riiid_data = riiid_data[riiid_data.rating != -1] # If you get rid of some rows, you risk an uneven split in split_loo


##### UNCOMMENT TO REFORMAT DATA ########
# riiid_dir = '../../input/riiid_train.feather'
# riiid_data = pd.read_feather(riiid_dir)

# print("finished data read")
# riiid_data.rename(columns={'answered_correctly':'rating'}, inplace=True)
# print(riiid_data.head())

# # Reindex
# user_id = riiid_data[['user_id']].drop_duplicates().reindex()
# user_id['userId'] = np.arange(len(user_id))
# riiid_data = pd.merge(riiid_data, user_id, on=['user_id'], how='left')
# print("users reindexed")
# content_id = riiid_data[['content_id']].drop_duplicates()
# content_id['itemId'] = np.arange(len(content_id))
# riiid_data = pd.merge(riiid_data, content_id, on=['content_id'], how='left')
# print("questions reindexed")
# riiid_data = riiid_data[['userId', 'itemId', 'rating', 'timestamp']]
# print('Range of userId is [{}, {}]'.format(riiid_data.userId.min(), riiid_data.userId.max()))
# print('Range of itemId is [{}, {}]'.format(riiid_data.itemId.min(), riiid_data.itemId.max()))
# riiid_data.to_feather("riiid_data.feather")
##########

riiid_data = pd.read_feather('../../input/riiid_data.feather')

print(riiid_data.info)

# DataLoader for training
sample_generator = SampleGenerator(ratings=riiid_data)
evaluate_data = sample_generator.evaluate_data
# Specify the exact model
config = gmf_config
engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)
# config = neumf_config
# engine = NeuMFEngine(config)
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)