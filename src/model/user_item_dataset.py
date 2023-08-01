import numpy as np
import random


class UserItemDataset:
    '''Representation of the user-item dataset

    Attributes:
        users_data (numpy.ndarray): (n, 1) array where rows are n user ids and columns are
                                    the gender of the users (0: male, 1: female).
        items_data (numpy.ndarray): (m, g) array where rows are m item ids and columns are
                                    the g possible genres.
                                    If an item i has the genre j, then items_data[i][j] == 1, 0 otherwise
        interactions_data (numpy.ndarray): (n, m) array where rows are n user ids and columns are
                                           m item ids.
                                           If a user i rated item j then interactions_data[i, j] == 1,
                                           0 otherwise.
        genres_mapping (dictionary): The mapping between genre textual representation and their ids.
                                    

    '''

    def __init__(self):
        self.users_data = None
        self.items_data = None
        self.interactions_data = None
        self.genres_mapping = dict()


    def load_data(self,
                  info_filepath,
                  users_filepath,
                  items_filepath,
                  genres_filepath,
                  ratings_filepath):
        '''Reads user-movie ratings in MovieLens format.

        Source: https://grouplens.org/datasets/movielens/

        '''

        # Read users and items number
        with open(info_filepath) as f:
            users_num = int(f.readline().split()[0])
            items_num = int(f.readline().split()[0])

        # Create users array
        self.users_data = np.zeros((users_num, 1), dtype=int)

        #
        # Load users data
        #   Column 0 -> Gender: 0 Male, 1 Female
        #
        with open(users_filepath) as f:
            for line in f:
                line_data = line.split('|')

                # Read gender
                if line_data[2] == 'M':
                    gender_value = 0
                elif line_data[2] == 'F':
                    gender_value = 1

                self.users_data[int(line_data[0]) - 1][0] = gender_value
        

        # Load genres
        with open(genres_filepath) as f:
            for line in f:
                line_data = line.split('|')

                self.genres_mapping[line_data[0]] = int(line_data[1])

        genres_num = len(self.genres_mapping)

        # Create items array
        self.items_data = np.zeros((items_num, genres_num), dtype=int)

        #
        # Load items data
        #   Columns -> genres (0/1)
        #
        with open(items_filepath) as f:
            for line in f:
                line_data = line.split('|')
                genres_data = [int(i) for i in line_data[-genres_num:]]

                self.items_data[int(line_data[0]) - 1] = np.asarray(genres_data)

        # Create interactions data array
        self.interactions_data = np.zeros((users_num, items_num), dtype=int)

        # Load interactions data
        with open(ratings_filepath) as f:
            for line in f:
                line_data = line.split()
                user_id, item_id = int(line_data[0]) - 1, int(line_data[1]) - 1

                self.interactions_data[user_id][item_id] = 1


    def filter_interactions_by_genre(self,
                                     genre_list,
                                     exclusive=False):
        '''Creates a subset of the interaction matrix based on a genres list

        Args:
            genre_list (list): A list with the literal of the genres (e.g. ["Action", "Adventure"])
            exclusive (boolean): If True, then each filtered item is allowed to belong to exactly one genre
        Returns:
            user_ids (numpy.ndarray): An array with the original user ids of the filtered users
            item_ids (numpy.ndarray): An array with the original item ids of the filtered items
            res (numpy.ndarray): The filtered subarray of the original interactions array
        '''
        # Get genre ids
        genre_ids = [self.genres_mapping[genre] for genre in genre_list]

        # Get item ids that have at least one of the genres
        item_ids = np.nonzero(np.any(self.items_data[:, genre_ids], axis=1))[0]

        # If exclusive=True, eliminate items that belong to more than one genre
        if exclusive:
            item_ids_to_remove = []
            for item_id in item_ids:
                count = 0

                for genre_id in genre_ids:
                    if self.items_data[item_id][genre_id] == 1:
                        count +=1

                if count > 1:
                    item_ids_to_remove.append(item_id)

            for item_id in item_ids_to_remove:
                item_ids = np.delete(item_ids, np.where(item_ids == item_id)[0][0])

        # Get users that have interacted with these items
        user_ids = np.nonzero(np.any(self.interactions_data[:, item_ids], axis=1))[0]

        # Get the subarray of the interactions
        res = self.interactions_data[user_ids, :]
        res = res[:, item_ids]

        return user_ids, item_ids, res


    def get_genres_by_id(self,
                         item_id):
        return self.items_data[item_id]


    def get_seen_items_by_user_id(self,
                                  user_id):
        return np.nonzero(self.interactions_data[user_id])


    def get_x_users_by_genre(self,
                             user_ids,
                             x):
        '''Returns x male and x female user ids.

        Args:
            user_ids (numpy.ndarray): An array with user ids to choose from.
            x (int): The number of male and female users to return
        Returns:
            male_ids (list): A list with the male ids
            female_ids (list): A list with female ids
        '''
        male_ids = []
        female_ids = []

        for user_id in user_ids:
            if self.users_data[user_id] == 0:
                male_ids.append(user_id)
            else:
                female_ids.append(user_id)

        return random.sample(male_ids, x), random.sample(female_ids, x)
        

    def is_genre(self,
                 item_id,
                 genre):
        if self.items_data[item_id][self.genres_mapping[genre]] == 1:
            return True
        else:
            return False


    def is_male(self,
                user_id):
        return True if self.users_data[user_id][0] == 0 else False


    def get_gender(self,
                   user_id):
        return self.users_data[user_id][0]
