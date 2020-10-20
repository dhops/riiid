import pandas as pd

riiid_dir = '../../input/riiid-test-answer-prediction/train.csv'
riiid_data = pd.read_csv(riiid_dir, sep=',', nrows=10**4, header=None, skiprows=1, names=['row_id', 'timestamp', 'user_id', 'content_id', 'content_type_id', 'task_container_id', 'user_answer', 'rating', 'prior_question_elapsed_time', 'prior_question_had_explanation'],  engine='python')

riiid_data
toy_train_df.to_csv('toy_train.csv')