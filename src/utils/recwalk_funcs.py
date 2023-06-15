import numpy as np


def create_recwalk_model_from_adj(adj_mat,
                                  alpha=1,
                                  gamma_1=1,
                                  gamma_2=1,
                                  C=0):
    '''Creates the transition probability matrix P from adjacency matrix

    Args:
        adj_mat (numpy.ndarray): The adjacency matrix of the graph
        a (float): Parameter for the weighted sum of matrices H and M in P
        gamma_1 (float): TODO: fill description
        gamma_2 (float): TODO: fill description
        C (int): TODO: fill description
    Returns:
        P (numpy.ndarray): The transition probability matrix
    '''
    # ------------ #
    # TODO: Matrix M calculation
    M = 0
    # ------------ #

    H = np.linalg.inv(np.diag(adj_mat @ np.ones(adj_mat.shape[0]))) @ adj_mat # Calculate matrix H
    
    P = alpha*H + (1-alpha)*M # Calculate transition probability matrix P
    
    return P


def create_recwalk_model(R,
                         alpha=1,
                         gamma_1=1,
                         gamma_2=1,
                         C=0,
                         remove_user_to_item=None,
                         single_direction=True,
                         rev=False):
    '''Creates the transition probability matrix P
    
    Args:
        R (numpy.ndarray): The user-item interaction matrix
        a (float): Parameter for the weighted sum of matrices H and M in P
        gamma_1 (float): TODO: fill description
        gamma_2 (float): TODO: fill description
        C (int): TODO: fill description
        remove_user_to_item (tuple): If not None, it's a list with tuples (user_id, item_id) to
                                     remove from adjacency matrix
        single_direction (boolean): If True, then edges defined in remove_user_to_item are removed for
                                    the direction user_id->item_id.
                                    If false, both directions are removed user_id->item_id and item_id->user_id
                                    TODO: Update description to go along with 'reversed' parameter.
        rev (boolean): If True, item_id->user_id direction is removed.
    Returns:
        P (numpy.ndarray): The transition probability matrix
    '''
    
    # ------------ #
    # TODO: Matrix M calculation
    M = 0
    # ------------ #
    
    users_num, items_num = R.shape[0], R.shape[1]
    
    # Create adjacency matrix for the bipartite graph
    adj_mat = np.zeros((users_num + items_num, users_num + items_num), dtype=int)
    adj_mat[:users_num, users_num:] = R
    adj_mat[users_num:, :users_num] = R.T

    # Remove edge from user to item
    if remove_user_to_item is not None:
        for tup in remove_user_to_item:
            user_id, item_id = tup[0], tup[1]
            # item_id = tup[1]

            if rev:
                adj_mat[users_num + item_id, user_id] = 0

                if single_direction is False: # Remove both directions
                    adj_mat[user_id, users_num + item_id] = 0
            else:
                adj_mat[user_id, users_num + item_id] = 0

                if single_direction is False: # Remove both directions
                    adj_mat[users_num + item_id, user_id] = 0
    
    # Calculate matrix H
    H = np.linalg.inv(np.diag(adj_mat @ np.ones(adj_mat.shape[0]))) @ adj_mat
    
    # Calculate transition probability matrix P
    P = alpha*H + (1-alpha)*M
    
    return P