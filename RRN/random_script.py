for k, v in tag_dict.items():
	if type(v) is list:
		if max(v)>m:
			m = max(v)
			ind = k
print(m)
print(ind)
print(tag_dict[ind])


used_ind = np.ones(188)
for k, v in tag_dict.items():
	if type(v) is list:
		for t in v:
			used_ind[t]=1




		if max(v)>m:
			m = max(v)
			ind = k
print(m)
print(ind)
print(tag_dict[ind])