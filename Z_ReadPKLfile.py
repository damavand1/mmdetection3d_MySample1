import pickle

#pkl file path
file_path = 'demo/data/kitti/000008.pkl'

#Open and readomg .pkl file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

#showing pkl file
print(data)



