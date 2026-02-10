# HGCF utils/data_generator.py 에 삽입할 odor 분기 (elif dataset == 'odor': 블록)
# 원본: " elif dataset.split('-')[0] in ['Amazon', 'yelp']:" 앞에 삽입.

ODOR_BRANCH = r''' elif dataset == 'odor':
 pkl_path = os.path.join('./data/', dataset)
 self.pkl_path = pkl_path
 self.dataset = dataset
 with open(os.path.join(pkl_path, 'train.pkl'), 'rb') as f:
  self.train_dict = pkl.load(f)
 with open(os.path.join(pkl_path, 'test.pkl'), 'rb') as f:
  self.test_dict = pkl.load(f)
 self.num_users = max(self.train_dict.keys()) + 1
 all_items = set()
 for items in list(self.train_dict.values()) + list(self.test_dict.values()):
  all_items.update(items)
 self.num_items = max(all_items) + 1 if all_items else 0
 self.adj_train, _ = self.generate_adj()
 if eval(norm_adj):
  self.adj_train_norm = normalize(self.adj_train + sp.eye(self.adj_train.shape[0]))
  self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)
 print('num_users %d, num_items %d' % (self.num_users, self.num_items))
 print('adjacency matrix shape: ', self.adj_train.shape)
 self.user_item_csr = self.generate_rating_matrix([*self.train_dict.values()], self.num_users, self.num_items)
'''
