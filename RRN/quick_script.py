import torch
from RRNCF.model import RRNCF

    
n_item = 13523
n_tag = 188
# n_users = 400000

model = RRNCF(embed_dim=16, mlp_dim=16, dropout=0.2, questionset_size=n_item, tagset_size=n_tag)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
print(sum(p.numel() for p in model.parameters()))

# import numpy as np
# for k,v in l_tag_map.items():
# 	if type(v[0]) != np.int64:
# 		print(k)
# 		print(v)