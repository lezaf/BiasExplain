import numpy         as np
import scipy.sparse  as sparse


def calculate_lambda(scores_absorbing_items,
                     user_x_id,
                     nb_ids,
                     nbs_to_remove,
                     genre_absorbing_col,
                     gamma):
    '''
    Args:
        scores_absorbing_items (numpy.2darray): Items absorbing scores         (p_j) 
        user_x_id (int): Mapped id of current user                         (u)
        nb_ids (numpy.1darray): Mapped ids of user_x_id user  (R_u)
        nbs_to_remove (numpy.1darray): Mapped ids of neighbors to remove (S_u)
        genre_absorbing_col (int): Column of I_0 in absorbing scores           (I_0)
        gamma (float): Gamma parameter
    '''
    # Calculate numerator
    numerator_avg_1 = np.sum(scores_absorbing_items[nb_ids, genre_absorbing_col])/nb_ids.shape[0]
    numerator_avg_2 = np.sum(scores_absorbing_items[nbs_to_remove, genre_absorbing_col])/nbs_to_remove.shape[0]
    numerator = ((1-gamma)/gamma)*(numerator_avg_1 - numerator_avg_2)

    # Calculate denominator
    denominator_avg_1 = np.sum(scores_absorbing_items[nb_ids, -1])/nb_ids.shape[0]
    denominator_avg_2 = np.sum(scores_absorbing_items[nbs_to_remove, -1])/nbs_to_remove.shape[0]
    denominator = (nb_ids.shape[0] - nbs_to_remove.shape[0])/nbs_to_remove.shape[0]
    denominator -= ((1-gamma)/gamma)*(denominator_avg_1 - denominator_avg_2)
    
    return numerator/denominator


def _calculate_scores(P_absorbing,
                      scores_prev,
                      tol):
    '''Calculates absorbing scores

    TODO:
        -Add iterations stop criterion
    '''
    loop_cnt = 0
    error = 1 # Initialize error to be greater than tol
    while error > tol:
        scores_cur = P_absorbing @ scores_prev # Calculate new scores

        error = np.linalg.norm((scores_cur - scores_prev), ord=1) # Calculate new difference

        scores_prev = scores_cur # Update scores

        loop_cnt += 1

    return scores_cur


def _construct_absorbing_array(P,
                               num_absorb,
                               gamma):
    P_absorbing = (1-gamma) * P
    P_absorbing = np.concatenate((P_absorbing, np.zeros((num_absorb, P_absorbing.shape[1]))), axis=0)
    P_absorbing = np.concatenate((P_absorbing, np.zeros((P_absorbing.shape[0], num_absorb))), axis=1)
    P_absorbing[-num_absorb:, -num_absorb:] = np.identity(num_absorb)

    return P_absorbing


def ppr_absorbing_5(P,
                    user_x_id,
                    neighbor_ids,
                    item_group_ids_0,
                    item_group_ids_1,
                    users_num,
                    gamma=0.15,
                    tol=0.00000000000001):
    '''Calculates ppr scores relative to a user-node of interest x.
    
    Args:
        P (numpy.ndarray): Transition probability matrix
        user_x_id (int): The mapped user id of the node of interest
        neighbor_ids (numpy.1darray): Neighbors of user_x_id node
        item_group_ids_0 (numpy.1darray): The item ids of I_0
        item_group_ids_1 (numpy.1darray): The item ids of I_1
        d_factor (float): The probability of restart (damping factor)
        tol (float): Convergence tolerance
    Returns:
        The PPR scores for every node
    '''
    # Example absorbing matrix for I_0='Action' and I_1='Romance':
    #
    #                                   a_A a_R a_U a_N a_x                 P_aA P_aR P_aU P_aN P_ax
    #               |                 #                     |             |  0    0    ...       0   |
    #               |                 #       gamma         |             | ...                      |
    #               |   (1-gamma)*P   #       where         |             |                          |
    #               |                 #       needed        |             |  0         ...       0   |
    # P_absorbing = |#######################################|    scores = |##########################|
    #               |0 0 ...        0 #  1   0   0   0   0  |             |  1    0    0    0    0   |
    #               |0 0 ...        0 #  0   1   0   0   0  |             |  0    1    0    0    0   |
    #               |0 0 ...        0 #  0   0   1   0   0  |             |  0    0    1    0    0   |
    #               |0 0 ...        0 #  0   0   0   1   0  |             |  0    0    0    1    0   |
    #               |0 0 ...        0 #  0   0   0   0   1  |             |  0    0    0    0    1   |
    
    # # Helper variable; number of absorbing nodes used
    num_absorb = 5
    
    # Construct the absorbing transition prob. matrix
    P_absorbing = _construct_absorbing_array(P=P, num_absorb=num_absorb, gamma=gamma)
    
    P_absorbing[np.setdiff1d(item_group_ids_0, neighbor_ids) + users_num, -5] = gamma # Link I_0 items (no neighbors) w/ a_(I_0)
    P_absorbing[np.setdiff1d(item_group_ids_1, neighbor_ids) + users_num, -4] = gamma # Link I_1 items (no neighbors) w/ a_(I_1)
    P_absorbing[np.setdiff1d(np.arange(users_num), user_x_id), -3] = gamma            # Link users (no user_x_id) w/ a_U
    P_absorbing[neighbor_ids + users_num, -2] = gamma                                 # Link neighbors w/ a_N
    P_absorbing[user_x_id, -1] = gamma                                                # Link current user w/ a_x

    P_absorbing = sparse.csr_array(P_absorbing) # Handle P_absorbing as sparse

    # Create scores matrix
    scores_prev = np.concatenate((np.zeros((P.shape[0], num_absorb)), np.identity(num_absorb)), axis=0)

    return _calculate_scores(P_absorbing, scores_prev, tol)


def ppr_absorbing_4(P,
                    user_x_id,
                    item_group_ids_0,
                    item_group_ids_1,
                    users_num,
                    gamma=0.15,
                    tol=0.00000000000001):
    '''Calculates ppr scores relative to a user-node of interest u.
    
    Args:
        TODO: Fill in
    Returns:
        The PPR scores for every node
    '''
    #                                   a_A a_R a_U a_x                  P_aA P_aR P_aU P_ax
    #               |                 #                  |             |  0    0    ...  0   |
    #               |                 #      gamma       |             | ...                 |
    #               |   (1-gamma)*P   #      where       |             |                     |
    #               |                 #      needed      |             |  0              0   |
    # P_absorbing = |####################################|    scores = |#####################|
    #               |0 0 ...        0 #  1   0   0   0   |             |  1    0    0    0   |
    #               |0 0 ...        0 #  0   1   0   0   |             |  0    1    0    0   |
    #               |...          ... #  0   0   1   0   |             |  0    0    1    0   |
    #               |0 0 ...        0 #  0   0   0   1   |             |  0    0    0    1   |
    
    # Helper variable; number of absorbing nodes used
    num_absorb = 4
    
    # Construct the transition prob. matrix using absorbing nodes
    P_absorbing = _construct_absorbing_array(P=P, num_absorb=num_absorb, gamma=gamma)

    P_absorbing[item_group_ids_0 + users_num, -4] = gamma # Link I_0 items w/ a_(I_0)
    P_absorbing[item_group_ids_1 + users_num, -3] = gamma # Link I_1 items w/ a_(I_1)
    P_absorbing[np.setdiff1d(np.arange(users_num), user_x_id), -2] = gamma # Link users (no user_x_id) w/ a_U
    P_absorbing[user_x_id, -1] = gamma # Link current user with a_x

    P_absorbing = sparse.csr_array(P_absorbing) # Handle P_absorbing as sparse
            
    # Initialize scores matrix
    scores_prev = np.concatenate((np.zeros((P.shape[0], num_absorb)), np.identity(num_absorb)), axis=0)

    return _calculate_scores(P_absorbing, scores_prev, tol)


# PPR with 3 absorbing nodes
def ppr_absorbing_3(P,
                    item_group_ids_0,
                    item_group_ids_1,
                    users_num,
                    gamma=0.15,
                    tol=0.00000000000001):
    '''Calculates ppr scores allocated from every node to Action, Romance and Users.
    
    Args:
        P (numpy.ndarray): Transition probability matrix
        my_dataset (UserItemDataset): The dataset object
        d_factor (float): The probability of restart (damping factor)
        tol (float): Convergence tolerance
    Returns:
        The PPR scores for every node
    '''
    #                                    a_A a_R a_U
    #               |                 #              |
    #               |                 #              |
    #               |   (1-gamma)*P   #     gamma    |
    #               |                 #              |
    # P_absorbing = |################################|
    #               |0 0 ...        0 #  1   0   0   |
    #               |0 0 ...        0 #  0   1   0   |
    #               |...          ... #  0   0   1   |
    
    # Helper variable; number of absorbing nodes used
    num_absorb = 3
    
    # Construct the transition prob. matrix using absorbing nodes
    P_absorbing = _construct_absorbing_array(P=P, num_absorb=num_absorb, gamma=gamma)

    P_absorbing[item_group_ids_0 + users_num, -3] = gamma # Link I_0 items w/ a_(I_0)
    P_absorbing[item_group_ids_1 + users_num, -2] = gamma # Link I_1 items w/ a_(I_1)
    P_absorbing[np.arange(users_num), -1] = gamma # Link users w/ a_U
            
    P_absorbing = sparse.csr_array(P_absorbing) # Handle P_absorbing as sparse

    # Initialize scores matrix
    scores_prev = np.concatenate((np.zeros((P.shape[0], num_absorb)), np.identity(num_absorb)), axis=0)

    return _calculate_scores(P_absorbing, scores_prev, tol)


def ppr_absorbing_2_item_side(P,
                              item_mapped_i,
                              users_num,
                              items_num,
                              gamma=0.15,
                              tol=0.00000000000001):
    '''Calculates PPR scores relative to item-node i.

                                       a_rest a_i                   P_a(rest) P_ai
                  |                 #              |             |                 |
                  |                 #     gamma    |             |                 |
                  |   (1-gamma)*P   #     where    |             |  PPR scores     |
                  |                 #     needed   |             |                 |
    P_absorbing = |################################|    scores = |#################|
                  |0 0 ...        0 #    1    0    |             |    1        0   |
                  |0 0 ...        0 #    0    1    |             |    0        1   |
    '''
    # Helper variable; number of absorbing nodes used
    num_absorb = 2

    # Construct the transition prob. matrix using absorbing nodes
    P_absorbing = _construct_absorbing_array(P=P, num_absorb=num_absorb, gamma=gamma)

    P_absorbing[np.arange(users_num), -2] = gamma # Link all users w/ a_rest
    P_absorbing[np.setdiff1d(np.arange(items_num), item_mapped_i) + users_num, -2] = gamma # Link rest w/ a_rest
    P_absorbing[users_num + item_mapped_i, -1] = gamma # Link current item w/ a_i

    P_absorbing = sparse.csr_array(P_absorbing) # Handle P_absorbing as sparse

    # Create scores matrix
    scores_prev = np.concatenate((np.zeros((P.shape[0], num_absorb)), np.identity(num_absorb)), axis=0)

    return _calculate_scores(P_absorbing, scores_prev, tol)
