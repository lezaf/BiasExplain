'''The file contains functions to make experiments on the Individual-level user-side part.
The experiments implemented here uses all PPR allocated to categories, both seen and unseen.
'''
import numpy                            as np
import time
import datetime

from   utils.recwalk_funcs              import create_recwalk_model, create_recwalk_model_from_adj
from   utils.counterfactual             import ppr_absorbing_4, calculate_lambda


def indiv_u_side_multiple_exp(R,
                              user_original_ids,
                              item_original_ids,
                              item_group_mapped_ids_0,
                              item_group_mapped_ids_1,
                              genre_absorbing_col,
                              directions_removed,
                              sample_users,
                              nbs_to_remove_limit=-1,
                              export_dir_path=None,
                              export_filename=None,
                              export_results=True):
    '''Calculate increase in PPR allocated from users in sample_users to a specified item category.
    In this experiment, ppr_absorbing_4 it's used, i.e. no distinction between seen/unseen is made.

    '''
    sample_size = sample_users.shape[0]
    single_direction = True if directions_removed == 1 else False
    users_num, items_num = R.shape[0], R.shape[1]
        
    # PPR scores allocated to unseen of genre in absorbing genre (column) 0 and 1 respectively
    ppr_0_all, ppr_1_all = [], []
    neighbor_mapped_ids_removed_all = []

    user_cnt = 0
    start_wall_t = time.time() # Keep time started
    for user_id in sample_users:
        ppr_0, ppr_1 = [], []
        deltas_0, deltas_1 = [], []

        nb_ids = np.nonzero(R[user_id])[0] # Find current user neighbors

        print(f"[{datetime.datetime.now()}] User {user_cnt + 1} - id: {user_original_ids[user_id]}:" +
                                            f"#neighbors={nb_ids.shape[0]}")

        # Create the model
        P = create_recwalk_model(R,
                                 single_direction=single_direction) # TODO: single_direction argument not needed since we have
                                                                    # one model only, while we use Equation (4)

        # Calculate scores using ppr_absorbing_4 once
        scores_absorbing = ppr_absorbing_4(P=P,
                                           user_x_id=user_id,
                                           item_group_ids_0=item_group_mapped_ids_0,
                                           item_group_ids_1=item_group_mapped_ids_1,
                                           users_num=users_num,
                                           gamma=0.15)

        # Keep the part of the array related to items
        scores_absorbing_items = scores_absorbing[users_num:(users_num+items_num)]

        # Add initial PPR unseen scores to lists
        ppr_0.append(scores_absorbing[user_id, 0])
        ppr_1.append(scores_absorbing[user_id, 1])

        # Define another variable to keep remaining neighbors after removal
        nb_ids_available = nb_ids.copy()
        nb_ids_removed = np.array([], dtype=int)
        nbs_removed_cnt = 0
        while True:
            nb_deltas_0, nb_deltas_1 = [], []

            # Calculate delta score for each neighbor
            for nb_to_remove in nb_ids_available:
                S_u = np.insert(nb_ids_removed, 0, nb_to_remove) # Create S_u adding candidate neighbor to remove

                # --- Equation (4) in paper - Efficient calculation of delta --- #
                d_0 = scores_absorbing[user_id, -1]*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
                                                                     user_x_id              = user_id,
                                                                     nb_ids                 = nb_ids,
                                                                     nbs_to_remove          = S_u,
                                                                     genre_absorbing_col    = 0,
                                                                     gamma                  = 0.15)

                d_1 = scores_absorbing[user_id, -1]*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
                                                                     user_x_id              = user_id,
                                                                     nb_ids                 = nb_ids,
                                                                     nbs_to_remove          = S_u,
                                                                     genre_absorbing_col    = 1,
                                                                     gamma                  = 0.15)

                # Store deltas
                nb_deltas_0.append(d_0)
                nb_deltas_1.append(d_1)

            # Pick maximum delta score neighbor depending on genre_absorbing_col
            nb_to_remove_ind = np.argmax(np.array(nb_deltas_0)) if genre_absorbing_col == 0 else np.argmax(np.array(nb_deltas_1))
            
            d_0, d_1 = nb_deltas_0[nb_to_remove_ind], nb_deltas_1[nb_to_remove_ind]
            cur_d = d_0 if genre_absorbing_col == 0 else d_1

            if cur_d <= 0: # If negative delta, stop
                print("Stop reason: negative delta score.")
                break

            # Store delta
            deltas_0.append(d_0)
            deltas_1.append(d_1)

            # Update available and removed lists
            nb_ids_removed = np.append(nb_ids_removed, nb_ids_available[nb_to_remove_ind])
            nb_ids_available = np.delete(nb_ids_available, nb_to_remove_ind)

            # Keep one neighbor
            if nb_ids_available.shape[0] == 1:
                break

            # Increment counter
            nbs_removed_cnt += 1
            
            if nbs_removed_cnt % 10 == 0: # Display progress message
                print(f"[{datetime.datetime.now()}] {nbs_removed_cnt} neighbors removed (refreshed per 10)")

            # Check if limit for how many neighbors to remove has been specified
            if nbs_to_remove_limit != -1:
                if nbs_removed_cnt == nbs_to_remove_limit: # If limit reached, stop
                    break

        # Calculate new PPR scores adding up delta scores
        for i in range(nb_ids_removed.shape[0]):
            ppr_0.append(ppr_0[0] + deltas_0[i])
            ppr_1.append(ppr_1[0] + deltas_1[i])

        # Store results
        ppr_0_all.append(ppr_0)
        ppr_1_all.append(ppr_1)
        neighbor_mapped_ids_removed_all.append(nb_ids_removed)
        
        user_cnt += 1
        print("----------")
        
    # Keep time ended
    end_wall_t = time.time()
    print(f"Wall time: {round((end_wall_t-start_wall_t)/60, 4)} minutes")

    ##### EXPORT RESULTS TO FILE #####
    if export_results:
        if export_dir_path is None:
            export_dir_path = './output/individual'
        if export_filename is None:
            export_filename = f"experiment11a_{sample_size}_R_{directions_removed}_mult"

        _export_to_file(sample_users,
                        sample_size,
                        ppr_0_all,
                        ppr_1_all,
                        neighbor_mapped_ids_removed_all,
                        user_original_ids,
                        item_original_ids,
                        export_dir_path,
                        export_filename)


def indiv_u_side_recalculation_exp(R,
                                   user_original_ids,
                                   item_original_ids,
                                   item_group_mapped_ids_0,
                                   item_group_mapped_ids_1,
                                   genre_absorbing_col,
                                   directions_removed,
                                   sample_users,
                                   export_dir_path=None,
                                   export_filename=None):
    sample_size = sample_users.shape[0]
    single_direction = True if directions_removed == 1 else False
    users_num, items_num = R.shape[0], R.shape[1]
    
    ppr_0_all, ppr_1_all = [], [] # PPR scores allocated to item groups 0 and 1 for all users
    nb_ids_removed_all = [] # Keep removed neighbor ids for all users

    user_cnt = 0
    start_wall_t = time.time()
    for user_id in sample_users:
        ppr_0, ppr_1 = [], []

        nb_ids = np.nonzero(R[user_id])[0]  # Find current user neighbors

        print(f"[{datetime.datetime.now()}] User {user_cnt + 1} - id: {user_original_ids[user_id]}:" +
                                            f"#neighbors={nb_ids.shape[0]}")

        # Create adjacency matrix
        adj_mat = np.zeros((users_num + items_num, users_num + items_num), dtype=int)
        adj_mat[:users_num, users_num:] = R
        adj_mat[users_num:, :users_num] = R.T

        # Define another variable to keep remaining neighbors after removal
        nb_ids_available = nb_ids.copy()
        nb_ids_removed = []  # Neighbor ids removed for current user
        nbs_removed_cnt = 0
        while True:
            P = create_recwalk_model_from_adj(adj_mat) # Create the model

            # Calculate scores using absorbing random walk with 4 nodes
            scores_absorbing = ppr_absorbing_4(P=P,
                                               user_x_id=user_id,
                                               item_group_ids_0=item_group_mapped_ids_0,
                                               item_group_ids_1=item_group_mapped_ids_1,
                                               users_num=users_num,
                                               gamma=0.15)

            ppr_0.append(scores_absorbing[user_id][0])
            ppr_1.append(scores_absorbing[user_id][1])

            if nb_ids_available.shape[0] == 1: # Keep 1 neighbor
                break

            # Calculate delta score for each neighbor
            nb_deltas = []
            for nb_to_remove in nb_ids_available:
                # Keep the part of the array related to items
                scores_absorbing_items = scores_absorbing[users_num:(users_num+items_num)]

                # --- Equation (3) in paper --- #
                delta = scores_absorbing[user_id][-1]*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
                                                                       user_x_id              = user_id,
                                                                       nb_ids                 = nb_ids_available,
                                                                       nbs_to_remove          = np.array([nb_to_remove]),
                                                                       genre_absorbing_col    = genre_absorbing_col,
                                                                       gamma=0.15)

                # Store delta score
                nb_deltas.append(delta)

            # Find neighbor with the greatest delta score
            nb_to_remove_ind = np.argmax(np.array(nb_deltas))
            max_delta_score = nb_deltas[nb_to_remove_ind]

            # If negative delta, stop
            if max_delta_score <= 0:
                print("Stop reason: negative delta score.")
                break

            # Remove item
            adj_mat[user_id, users_num + nb_ids_available[nb_to_remove_ind]] = 0
            if not single_direction: adj_mat[users_num + nb_ids_available[nb_to_remove_ind], user_id] = 0

            nb_ids_removed.append(nb_ids_available[nb_to_remove_ind])
            nb_ids_available = np.delete(nb_ids_available, nb_to_remove_ind)

            # Display progress message
            nbs_removed_cnt += 1
            if nbs_removed_cnt % 10 == 0:
                print(f"[{datetime.datetime.now()}] {nbs_removed_cnt} neighbors removed. (refreshed per 10)")

        # Store results in list
        ppr_0_all.append(ppr_0)
        ppr_1_all.append(ppr_1)
        nb_ids_removed_all.append(np.array(nb_ids_removed))

        user_cnt += 1
        print("----------")
            
    # Keep time ended
    end_wall_t = time.time()
    print(f"Wall time: {round((end_wall_t-start_wall_t)/60, 4)} minutes")
            
    ##### EXPORT RESULTS TO FILE #####
    if export_dir_path is None:
        export_dir_path = './output/individual'
    if export_filename is None:
        export_filename = f"experiment11c_{sample_size}_R_{directions_removed}_recalculation"

    _export_to_file(sample_users,
                    sample_size,
                    ppr_0_all,
                    ppr_1_all,
                    nb_ids_removed_all,
                    user_original_ids,
                    item_original_ids,
                    export_dir_path,
                    export_filename)


def indiv_u_side_delta_once_exp(R,
                                user_original_ids,
                                item_original_ids,
                                item_group_mapped_ids_0,
                                item_group_mapped_ids_1,
                                genre_absorbing_col,
                                directions_removed,
                                sample_users,
                                export_dir_path=None,
                                export_filename=None):
    sample_size = sample_users.shape[0]
    single_direction = True if directions_removed == 1 else False
    users_num, items_num = R.shape[0], R.shape[1]

    ppr_0_all, ppr_1_all = [], [] # PPR scores allocated to item groups 0 and 1 for all users
    nb_ids_removed_all = []

    user_cnt = 0
    wall_t_elapsed = 0 # Overall experiment time
    for user_id in sample_users:
        start_wall_t = time.time() # Actual calculations time

        ppr_0, ppr_1 = [], []
        deltas_0, deltas_1 = [], []
        
        # Find neighbors of current user
        nb_ids = np.nonzero(R[user_id])[0]

        print(f"[{datetime.datetime.now()}] User {user_cnt + 1} - id: {user_original_ids[user_id]}:" +
                                            f"#neighbors={nb_ids.shape[0]}")

        # Create the model
        P = create_recwalk_model(R)

        # Calculate scores using absorbing random walk with 4 nodes
        scores_absorbing = ppr_absorbing_4(P=P,
                                           user_x_id=user_id,
                                           item_group_ids_0=item_group_mapped_ids_0,
                                           item_group_ids_1=item_group_mapped_ids_1,
                                           users_num=users_num,
                                           gamma=0.15)

        # Store initial PPR score allocated to category items
        ppr_0.append(scores_absorbing[user_id][0])
        ppr_1.append(scores_absorbing[user_id][1])

        # Calculate delta for all neighbors once
        for nb_to_remove in nb_ids:
            # Keep the part of the array related to items
            scores_absorbing_items = scores_absorbing[users_num:(users_num+items_num)]

            d_0 = scores_absorbing[user_id][-1]*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
                                                                 user_x_id              = user_id,
                                                                 nb_ids                 = nb_ids,
                                                                 nbs_to_remove          = np.array([nb_to_remove]),
                                                                 genre_absorbing_col    = 0,
                                                                 gamma                  = 0.15)

            d_1 = scores_absorbing[user_id][-1]*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
                                                                 user_x_id              = user_id,
                                                                 nb_ids                 = nb_ids,
                                                                 nbs_to_remove          = np.array([nb_to_remove]),
                                                                 genre_absorbing_col    = 1,
                                                                 gamma                  = 0.15)

            # Store delta scores
            deltas_0.append(d_0)
            deltas_1.append(d_1)

        # Pick neighbors depending on genre_absorbing_col
        if genre_absorbing_col == 0:
            sort_inds = np.argsort(np.array(deltas_0))[::-1]
            cur_d_scores = deltas_0
        else:
            sort_inds = np.argsort(np.array(deltas_1))[::-1]
            cur_d_scores = deltas_1

        end_wall_t = time.time() # Keep time ended
        wall_t_elapsed += (end_wall_t - start_wall_t) # Keep time elapsed per user (without timing actual scores calculation)

        # Create adjacency matrix
        adj_mat = np.zeros((users_num + items_num, users_num + items_num), dtype=int)
        adj_mat[:users_num, users_num:] = R
        adj_mat[users_num:, :users_num] = R.T

        nb_ids_removed = []
        for sort_ind in sort_inds[:-1]:
            if cur_d_scores[sort_ind] < 0: break

            nb_id_to_remove = nb_ids[sort_ind]

            adj_mat[user_id, users_num + nb_id_to_remove] = 0 # Remove user->item direction
            if not single_direction: adj_mat[users_num + nb_id_to_remove, user_id] = 0 # Remove item->user direction

            nb_ids_removed.append(nb_id_to_remove)

            P = create_recwalk_model_from_adj(adj_mat)
            scores_absorbing = ppr_absorbing_4(P=P,
                                               user_x_id=user_id,
                                               item_group_ids_0=item_group_mapped_ids_0,
                                               item_group_ids_1=item_group_mapped_ids_1,
                                               users_num=users_num,
                                               gamma=0.15)

            ppr_0.append(scores_absorbing[user_id][0])
            ppr_1.append(scores_absorbing[user_id][1])

        # Store results in lists
        ppr_0_all.append(ppr_0)
        ppr_1_all.append(ppr_1)
        nb_ids_removed_all.append(np.array(nb_ids_removed))
        
        user_cnt += 1
        print("----------")
    
    print(f"[{datetime.datetime.now()}] Wall time (excluding actual PPR calculations): {round(wall_t_elapsed/60, 4)} minutes")

    ##### EXPORT RESULTS TO FILE #####
    if export_dir_path is None:
        export_dir_path = './output/individual'
    if export_filename is None:
        export_filename = f"experiment11b_{sample_size}_R_single"

    _export_to_file(sample_users,
                    sample_size,
                    ppr_0_all,
                    ppr_1_all,
                    nb_ids_removed_all,
                    user_original_ids,
                    item_original_ids,
                    export_dir_path,
                    export_filename)

# TODO:
#   - Remove sample_size from arguments and take it from sample_users.shape[0]
def _export_to_file(sample_users,
                    sample_size,
                    ppr_0_all,
                    ppr_1_all,
                    nb_ids_removed_all,
                    user_original_ids,
                    item_original_ids,
                    export_dir_path,
                    export_filename):
    # NOTE: user and item ids are exported  from 0..(n-1)
    with open(f"{export_dir_path}/{export_filename}.tsv", 'w') as f:
        # Write header lines
        f.write('# File format (per 4 lines):\n')
        f.write('# <user_id>\n')
        f.write('# <p_u_genre_0_initial> <p_u_genre_0_new_0> <p_u_genre_0_new_1> [...] <p_u_genre_0_new_n>\n')
        f.write('# <p_u_genre_1_initial> <p_u_genre_1_new_0> <p_u_genre_1_new_1> [...] <p_u_genre_1_new_n>\n')
        f.write('# <item_id_removed_0> <item_id_removed_1> [...] <item_id_removed_n>\n')
        
        for i in range(sample_size):
            user_mapped_id = sample_users[i]
            ppr_0 = ppr_0_all[i]
            ppr_1 = ppr_1_all[i]
            neighbor_mapped_ids_removed = nb_ids_removed_all[i].tolist()
            
            # Write user_id
            f.write(str(user_original_ids[user_mapped_id]) + '\n')
            # Write scores for category 0
            for i in range(len(ppr_0)):
                if i != (len(ppr_0) - 1):
                    f.write(str(ppr_0[i]) + '\t')
                else:
                    f.write(str(ppr_0[i]) + '\n')
            # Write scores for category 1
            for i in range(len(ppr_1)):
                if i != (len(ppr_1) - 1):
                    f.write(str(ppr_1[i]) + '\t')
                else:
                    f.write(str(ppr_1[i]) + '\n')
            # Write neighbor original ids
            for i in range(len(neighbor_mapped_ids_removed)):
                if i != (len(neighbor_mapped_ids_removed) - 1):
                    f.write(str(item_original_ids[neighbor_mapped_ids_removed[i]]) + '\t')
                else:
                    f.write(str(item_original_ids[neighbor_mapped_ids_removed[i]]))
            if user_mapped_id != sample_users[-1]:
                f.write('\n')
            f.flush()


def indiv_u_side_exp4(R,
                      user_original_ids,
                      item_original_ids,
                      item_group_mapped_ids_0,
                      item_group_mapped_ids_1,
                      genre_absorbing_col,
                      sample_users,
                      export_dir_path=None,
                      export_filename=None):
    '''Calculates deltas for all neighbors of users in sample_users.
    
    Args:
        TODO: fill in
    '''
    sample_size = sample_users.shape[0]
    users_num, items_num = R.shape[0], R.shape[1]
    
    ppr_initial_all = []
    deltas_all = []
    nb_ids_all = []

    user_cnt = 0
    start_wall_t = time.time()
    for user_id in sample_users:
        deltas = []
        nb_ids = np.nonzero(R[user_id])[0]

        print(f"[{datetime.datetime.now()}] User {user_cnt + 1} - id: {user_original_ids[user_id]}:" +
                                            f"#neighbors={nb_ids.shape[0]}")

        P = create_recwalk_model(R)

        # Calculate scores using ppr_absorbing_4 once
        scores_absorbing = ppr_absorbing_4(P=P,
                                           user_x_id=user_id,
                                           item_group_ids_0=item_group_mapped_ids_0,
                                           item_group_ids_1=item_group_mapped_ids_1,
                                           users_num=users_num,
                                           gamma=0.15)

        ppr_initial_all.append(scores_absorbing[user_id][genre_absorbing_col]) # Keep initial score
        
        scores_absorbing_items = scores_absorbing[users_num:(users_num+items_num)] # Keep scores for the items
        for nb_to_remove in nb_ids:
            deltas.append(scores_absorbing[user_id][-1]*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
                                                                         user_x_id              = user_id,
                                                                         nb_ids                 = nb_ids,
                                                                         nbs_to_remove          = np.array([nb_to_remove]),
                                                                         genre_absorbing_col    = genre_absorbing_col,
                                                                         gamma                  = 0.15))
                
        deltas_all.append(deltas) # Store deltas
        nb_ids_all.append(nb_ids) # Store neighbor ids

        user_cnt += 1

    # Keep time ended
    end_wall_t = time.time()
    print(f"Wall time: {round((end_wall_t-start_wall_t)/60, 4)} minutes")
        
    ##### EXPORT RESULTS TO FILE #####
    if export_dir_path is None:
        export_dir_path = './output/individual/experiment4'
    if export_filename is None:
        export_filename = f"experiment4_{sample_size}"
        
    with open(f"{export_dir_path}/{export_filename}.tsv", 'w') as f:
        # Write header lines
        f.write('# File format (per 4 lines):\n')
        f.write('# <user_id>\n')
        f.write('# <initial_ppr>\n')
        f.write('# <delta_0> <delta_1> [...] <delta_n>\n')
        f.write('# <item_id_0> <item_id_1> [...] <item_id_n>\n')
        
        # Write data
        for i in range(len(sample_users)):
            # Data to write
            user_id = sample_users[i]
            ppr_initial = ppr_initial_all[i]
            deltas = deltas_all[i]
            nb_ids = nb_ids_all[i]
            
            f.write(f"{user_original_ids[user_id]}\n")
            f.write(f"{ppr_initial}\n")

            # Write delta scores
            for j in range(len(deltas)): f.write(f"{deltas[j]}\t") if j != (len(deltas) - 1) else f.write(f"{deltas[j]}\n")

            # Write neighbor original ids
            for j in range(len(nb_ids)): f.write(f"{item_original_ids[nb_ids[j]]}\t") if j != (len(nb_ids) - 1) else f.write(f"{item_original_ids[nb_ids[j]]}")
            
            if i != (len(sample_users) - 1):
                f.write('\n')