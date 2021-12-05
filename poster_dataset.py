import torch
from PIL import Image # for viewing images
import os
import numpy as np

class PosterDataset(torch.utils.data.Dataset):
    'Characterizes a dataset of poster images, labeled by their average rating on Letterboxd'
    def __init__(self, data_folder, full_db, indices, transform):
        'Initialization'
        self.labels = movies_full_df['boxd_vote_average']
        self.data_folder = data_folder
        self.full_db = full_db # movies_full_df
        self.movie_indices = list(indices) # this maps 0 -> train_index[0]
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.movie_indices)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        movie_index = self.movie_indices[index]
        if (type(movie_index) == list):
            return self.get_subset(movie_index)
        
        # Load data and get label. 
        X = self.transform(self.get_poster(movie_index))
        return X, np.float32(self.labels[movie_index])

    # get the actual, non-transformed, non-tensorized version of the image given index
    def get_poster(self, movie_index):
        return Image.open(self.get_poster_path(movie_index)).convert('RGB')

    def _get_poster_path(self, movie_index):
        movie_id = self.full_db['movie_id'][movie_index]
        folder_num = self.full_db['poster_path'][movie_index]
        img_name = movie_id + '.jpg'
        return os.path.join(self.data_folder, str(folder_num), img_name)

    def _get_subset(self, movie_indices):
        'Returns the images and labels in the given list of indices'
        images = [
            self.transform(self.get_poster(movie_index))
            for movie_index in movie_indices
        ]
        
        labels = [np.float32(self.labels[movie_index]) for movie_index in movie_indices]
        return images.float(), labels