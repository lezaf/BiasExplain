import numpy              as np
import networkx           as nx
import seaborn            as sns
import matplotlib.pyplot  as plt
import random
import time
import datetime
import os

from   utils.recwalk_funcs              import create_recwalk_model, create_recwalk_model_from_adj
from   utils.counterfactual             import ppr_absorbing_2_item_side
from   model.user_item_dataset          import UserItemDataset
from   misc.dataset_stats               import plot_statistics


# Remove FROM items x TO users
def individual_item_side_exp1(R,
                              user_original_ids,
                              item_original_ids,
                              sample_item_i_ids,
                              item_ids_all,
                              u_ids_0,
                              u_ids_1,
                              num_x_to_remove=-1,
                              choose_by_delta=True, # If True, removal based on delta, else on ratio
                              export_dir_path=None,
                              export_filename=None,
                              export_results=True):
    users_num, items_num = R.shape[0], R.shape[1]
    
    p_U_0_to_i_initial_all, p_U_1_to_i_initial_all = [], []
    p_U_0_to_i_new_all, p_U_1_to_i_new_all = [], []
    ratios_0_all = [] # We keep, for each item x, the ratio p_{U_0}(x)/p_{U_1}(x)
    item_x_ids_removed_all = []
    deltas_0_all = []
    
    P = create_recwalk_model(R) # Create transition prob. matrix
    ppr_u_to_i = calculate_matrix_Q(P)[:users_num, users_num:] # From Q, keep users to items scores part
    
    item_cnt = 0
    # Keep time started
    start_wall_t = time.time()
    for item_i_id in sample_item_i_ids:
        print(f"[{datetime.datetime.now()}] Item {item_cnt + 1} - id: {item_original_ids[item_i_id]}")
        
        p_U_0_to_i_initial_all.append(np.mean(ppr_u_to_i[u_ids_0, item_i_id]))
        p_U_1_to_i_initial_all.append(np.mean(ppr_u_to_i[u_ids_1, item_i_id]))
        
        item_x_ids = np.setdiff1d(item_ids_all, item_i_id) # All items except for item i
        
        # Precompute neighbors in U_0 and U_1
        x_nbs_all = [np.nonzero(R[:, x_id])[0] for x_id in item_x_ids]
        x_nbs_in_U_0_all = [np.intersect1d(u_ids_0, x_nbs) for x_nbs in x_nbs_all]
        
        print(f"[{datetime.datetime.now()}] Total items x: {item_x_ids.shape[0]}")
        
        deltas_0 = []
        ratios_0 = []
        invalid_x_inds = []
        for x_ind in range(item_x_ids.shape[0]): # Calculate deltas once
            cur_x_id = item_x_ids[x_ind]
        
            p_U_0_to_x = np.mean(ppr_u_to_i[u_ids_0, cur_x_id])
            p_U_1_to_x = np.mean(ppr_u_to_i[u_ids_1, cur_x_id])

            cur_x_nbs = x_nbs_all[x_ind]
            cur_x_nbs_in_U_0 = x_nbs_in_U_0_all[x_ind]

            # Calculate deltas
            d_term_2 = calculate_lambda_item_side(scores=ppr_u_to_i,
                                                  x_param=cur_x_id,
                                                  i_param=item_i_id,
                                                  S_x=cur_x_nbs_in_U_0,
                                                  R_x=cur_x_nbs)
            
            if d_term_2 < 0: # if negative lambda, x will decrease ppr
                invalid_x_inds.append(x_ind)
                continue
            
            deltas_0.append(p_U_0_to_x*d_term_2)            
            ratios_0.append(p_U_0_to_x/p_U_1_to_x)
        
        # Remove invalid indices
        x_nbs_all = np.delete(x_nbs_all, invalid_x_inds)
        x_nbs_in_U_0_all = np.delete(x_nbs_in_U_0_all, invalid_x_inds)
        item_x_ids = np.delete(item_x_ids, invalid_x_inds)
            
        # Create adjacency matrix
        adj_mat = np.zeros((users_num + items_num, users_num + items_num), dtype=int)
        adj_mat[:users_num, users_num:] = R
        adj_mat[users_num:, :users_num] = R.T
        
        # Order by delta/ratio, remove and recalculate
        sort_inds = np.argsort(deltas_0)[::-1] if choose_by_delta else np.argsort(ratios_0)[::-1]
        
        if num_x_to_remove > 0: sort_inds = sort_inds[:num_x_to_remove] # Num x best items to remove
        
        p_U_0_to_i_new_cur, p_U_1_to_i_new_cur = [], []
        for ind in sort_inds:            
            nbs_to_remove_ids_cur = x_nbs_in_U_0_all[ind]
            
            adj_mat[users_num + item_x_ids[ind], nbs_to_remove_ids_cur] = 0 # Remove item->user edges

            P = create_recwalk_model_from_adj(adj_mat)
            ppr_u_to_i_new = calculate_matrix_Q(P)[:users_num, users_num:]
            
            p_U_0_to_i_new_cur.append(np.mean(ppr_u_to_i_new[u_ids_0, item_i_id]))
            p_U_1_to_i_new_cur.append(np.mean(ppr_u_to_i_new[u_ids_1, item_i_id]))
            
            if (np.where(sort_inds == ind)[0][0] + 1) % 20 == 0: print(f"Counter: {np.where(sort_inds == ind)[0][0] + 1}")
        
        deltas_0_all.append(np.array(deltas_0)[sort_inds])
        ratios_0_all.append(np.array(ratios_0)[sort_inds])
        p_U_0_to_i_new_all.append(p_U_0_to_i_new_cur)
        p_U_1_to_i_new_all.append(p_U_1_to_i_new_cur)
        item_x_ids_removed_all.append(item_x_ids[sort_inds])
        
        item_cnt += 1
        
    # Keep time ended
    end_wall_t = time.time()
    print(f"Wall time: {round((end_wall_t-start_wall_t)/60, 4)} minutes")
        
    ##### EXPORT RESULTS TO FILE #####
    if export_results:
        file_exists = os.path.exists(f"{export_dir_path}/{export_filename}.tsv")
        with open(f"{export_dir_path}/{export_filename}.tsv", 'a') as f:
            if not file_exists:
                # Write header
                f.write("#File format (per 7 lines):\n")
                f.write("#<Item_id_i>\n")
                f.write("#<p_U_0_i_initial> <p_U_1_i_initial>\n")
                f.write("#<U_0_delta_0> <U_0_delta_1> [...] <U_0_delta_n>\n")
                f.write("#<p_U_0_i_new_0> <p_U_0_i_new_1> [...] <p_U_0_i_new_n>\n")
                f.write("#<p_U_1_i_new_0> <p_U_1_i_new_1> [...] <p_U_1_i_new_n>\n")
                f.write("#<item_id_x_removed_0> <item_id_x_removed_1> [...] <item_id_x_removed_n>\n")
                f.write("#<Group_0_ratio_0> <Group_0_ratio_1> [...] <Group_0_ratio_n>\n")
            else:
                f.write('\n')
            
            # Write results
            for i in range(sample_item_i_ids.shape[0]):
                item_original_id_cur = item_original_ids[sample_item_i_ids[i]]
                p_U_0_to_i_initial = p_U_0_to_i_initial_all[i]
                p_U_1_to_i_initial = p_U_1_to_i_initial_all[i]
                deltas_0 = deltas_0_all[i]
                ratios_0 = ratios_0_all[i]
                p_U_0_to_i_new = p_U_0_to_i_new_all[i]
                p_U_1_to_i_new = p_U_1_to_i_new_all[i]
                item_x_ids_removed = item_x_ids_removed_all[i]
                
                f.write(f"{item_original_id_cur}\n") # Write item id
                f.write(f"{p_U_0_to_i_initial}\t{p_U_1_to_i_initial}\n")
                
                for j in range(len(deltas_0)): f.write(f"{deltas_0[j]}\t") if j != (len(deltas_0) - 1) else f.write(f"{deltas_0[j]}\n")
                for j in range(len(p_U_0_to_i_new)): f.write(f"{p_U_0_to_i_new[j]}\t") if j != (len(p_U_0_to_i_new) - 1) else f.write(f"{p_U_0_to_i_new[j]}\n")
                for j in range(len(p_U_1_to_i_new)): f.write(f"{p_U_1_to_i_new[j]}\t") if j != (len(p_U_1_to_i_new) - 1) else f.write(f"{p_U_1_to_i_new[j]}\n")
                for j in range(len(item_x_ids_removed)): f.write(f"{item_original_ids[item_x_ids_removed][j]}\t") if j != (len(item_x_ids_removed) - 1) else f.write(f"{item_original_ids[item_x_ids_removed][j]}\n")
                for j in range(len(ratios_0)): f.write(f"{ratios_0[j]}\t") if j != (len(ratios_0) - 1) else f.write(f"{ratios_0[j]}")
                if i != (sample_item_i_ids.shape[0] - 1): f.write('\n')


# Remove FROM items x TO users with delta calculation ONCE
def individual_item_side_exp2(R,
                              item_mapped_id_i,
                              sample_items_x,
                              num_best_neighbors,
                              u_mapped_ids, # User neighbors mapped ids to remove edges to
                              u_mapped_ids_0,
                              export_dir_path=None,
                              export_filename=None,
                              export_results=True):
    users_num, items_num = R.shape[0], R.shape[1]
    
    P = create_recwalk_model(R) # Create transition prob. matrix
    ppr_users_to_items = calculate_matrix_Q(P)[:users_num, users_num:] # Calculate scores
    
    nbs_to_remove_all = []
    
    item_cnt = 0
    # Keep time started
    start_wall_t = time.time()
    for item_mapped_id_x in sample_items_x:
        print(f"[{datetime.datetime.now()}] Item {item_cnt + 1} - id: {item_original_ids[item_mapped_id_x]}")
        
        nb_ids = np.intersect1d(np.nonzero(R[:, item_mapped_id_x])[0], u_mapped_ids)
        if nb_ids.shape[0] < num_best_neighbors:
            print('Not enough neighbors.')
            return
        
        deltas = []
        d_term_1 = np.sum(ppr_users_to_items[u_mapped_ids_0, item_mapped_id_x])/u_mapped_ids_0.shape[0]
        for nb in nb_ids:
            d_term_2 = calculate_lambda_item_side(scores=ppr_users_to_items,
                                                  x_param=item_mapped_id_x,
                                                  i_param=item_mapped_id_i,
                                                  S_x=np.array([nb]),
                                                  R_x=nb_ids)
            deltas.append(d_term_1*d_term_2)
        
        nbs_to_remove_all.append(nb_ids[np.argsort(deltas)[::-1][:num_best_neighbors]])
        
        item_cnt += 1
        
    nbs_to_remove_tuples = [(j, sample_items_x[i]) for i in range(sample_items_x.shape[0]) for j in nbs_to_remove_all[i]]
    
    # Calculate initial PPR
    u_mapped_ids_1 = np.setdiff1d(np.arange(users_num), u_mapped_ids_0)
    ppr_scores_U_0_to_i = [np.sum(ppr_users_to_items[u_mapped_ids_0, item_mapped_id_i])/u_mapped_ids_0.shape[0]]
    ppr_scores_U_1_to_i = [np.sum(ppr_users_to_items[u_mapped_ids_1, item_mapped_id_i])/u_mapped_ids_1.shape[0]]
    
    # Create adjacency matrix
    adj_mat = np.zeros((users_num + items_num, users_num + items_num), dtype=int)
    adj_mat[:users_num, users_num:] = R
    adj_mat[users_num:, :users_num] = R.T
    
    # Remove and recalculate
    for i in range(len(nbs_to_remove_tuples)):
        edge = nbs_to_remove_tuples[i]
        adj_mat[users_num + edge[1], edge[0]] = 0 # Remove item->user direction
        
        P = create_recwalk_model_from_adj(adj_mat)
        ppr_users_to_items_new = calculate_matrix_Q(P)[:users_num, users_num:]
        
        ppr_scores_U_0_to_i.append(np.sum(ppr_users_to_items_new[u_mapped_ids_0, item_mapped_id_i])/u_mapped_ids_0.shape[0])
        ppr_scores_U_1_to_i.append(np.sum(ppr_users_to_items_new[u_mapped_ids_1, item_mapped_id_i])/u_mapped_ids_1.shape[0])
        
        if (i+1) % 20 == 0: print(f"[{datetime.datetime.now()}] {i+1} neighbors removed")
    
    # Keep time ended
    end_wall_t = time.time()
    print(f"Wall time: {round((end_wall_t-start_wall_t)/60, 4)} minutes")
    
    ##### EXPORT RESULTS TO FILE #####
    if export_results:
        file_exists = os.path.exists(f"{export_dir_path}/{export_filename}.tsv")
        with open(f"{export_dir_path}/{export_filename}.tsv", 'a') as f:
            if not file_exists:
                # Write header
                f.write("#File format (per 4 lines):\n")
                f.write("<num_central_items> <num_best_neighbors>\n")
                f.write("<p_U_0_to_i_initial> <p_U_0_to_i_new_0> <p_U_0_to_i_new_1> [...] <p_U_0_to_i_new_n>\n")
                f.write("<p_U_1_to_i_initial> <p_U_1_to_i_new_0> <p_U_1_to_i_new_1> [...] <p_U_1_to_i_new_n>\n")
                f.write("<user_id_removed_0> <user_id_removed_1> [...] <user_id_removed_n>\n")
            else:
                f.write('\n')
            
            f.write(f"{sample_items_x.shape[0]} {num_best_neighbors}\n")
            
            for i in range(len(ppr_scores_U_0_to_i)):
                if i != (len(ppr_scores_U_0_to_i) - 1):
                    f.write(f"{ppr_scores_U_0_to_i[i]}\t")
                else:
                    f.write(f"{ppr_scores_U_0_to_i[i]}\n")
            
            for i in range(len(ppr_scores_U_1_to_i)):
                if i != (len(ppr_scores_U_1_to_i) - 1):
                    f.write(f"{ppr_scores_U_1_to_i[i]}\t")
                else:
                    f.write(f"{ppr_scores_U_1_to_i[i]}\n")
            
            for i in range(len(nbs_to_remove_tuples)):
                if i != (len(nbs_to_remove_tuples) - 1):
                    f.write(f"{user_original_ids[nbs_to_remove_tuples[i][0]]}\t")
                else:
                    f.write(f"{user_original_ids[nbs_to_remove_tuples[i][0]]}")
