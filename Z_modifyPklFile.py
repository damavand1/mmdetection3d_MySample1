import pickle

# pkl file path
file_path = 'demo/data/kitti/000008.pkl'

# Open and reading .pkl file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Modify width and height
new_width = 640
new_height = 360

# Changing width and height in CAM2 section
data['data_list'][0]['images']['CAM2']['width'] = new_width
data['data_list'][0]['images']['CAM2']['height'] = new_height

# Saving modified data back to the same file
with open(file_path, 'wb') as f:
    pickle.dump(data, f)

print("Width and height changed successfully!")