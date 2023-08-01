'''The file contains group-group experiments.
'''
import numpy              as np
import networkx           as nx
import time
import datetime
import random
from   math               import ceil

from   utils.recwalk_funcs              import create_recwalk_model, create_recwalk_model_from_adj
from   utils.counterfactual             import ppr_absorbing_4, ppr_absorbing_3, calculate_lambda


def remove_best_k(R,
                  user_original_ids,
                  item_original_ids,
                  user_group_mapped_ids,
                  item_group_mapped_ids_0,
                  item_group_mapped_ids_1,
				  sample_users,
				  k_nbs,
				  genre_absorbing_col,
				  nbs_per_round=1,
				  export_dir_path=None,
                  export_filename=None,
                  export_results=True):
	'''Calculates the increase in p_{U_0}(I_0) when removing best k_nbs neighbors
	from each user in sample_users. The order of removal is decided by the overall
	order of sample_users_size*k_nbs edges.
	'''
	users_num, items_num = R.shape[0], R.shape[1]
	sample_size = sample_users.shape[0]

	deltas_0_all, deltas_1_all = [], []
	nbs_to_remove_all = []
	ppr_0, ppr_1 = [], []

	P = create_recwalk_model(R) # Create the model

	user_cnt = 0
	start_wall_t = time.time() # Keep time started
	for user_id in sample_users:
		nb_ids = np.nonzero(R[user_id])[0] # Find current user neighbors

		print(f"[{datetime.datetime.now()}] User {user_cnt + 1} - id: {user_original_ids[user_id]}:" +
                                            f"#neighbors={nb_ids.shape[0]}")

	    # Calculate scores using ppr_absorbing_4 once
		scores_absorbing = ppr_absorbing_4(P                = P,
										   user_x_id        = user_id,
										   item_group_ids_0 = item_group_mapped_ids_0,
										   item_group_ids_1 = item_group_mapped_ids_1,
										   users_num        = users_num,
										   gamma            = 0.15)

		scores_absorbing_items = scores_absorbing[users_num:(users_num+items_num)]

		d_term_1 = np.mean(scores_absorbing[user_group_mapped_ids, -1])
		deltas_0, deltas_1 = [], []
		for nb_to_remove in nb_ids:
			deltas_0.append(d_term_1*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
		                                              user_x_id              = user_id,
		                                              nb_ids                 = nb_ids,
		                                              nbs_to_remove          = np.array([nb_to_remove]),
		                                              genre_absorbing_col    = 0,
		                                              gamma                  = 0.15))

			deltas_1.append(d_term_1*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
		                                              user_x_id              = user_id,
		                                              nb_ids                 = nb_ids,
		                                              nbs_to_remove          = np.array([nb_to_remove]),
		                                              genre_absorbing_col    = 1,
		                                              gamma                  = 0.15))

        # Decide which category to increase based on genre_absorbing_col value
		sort_inds = np.argsort(deltas_0)[::-1][:k_nbs] if genre_absorbing_col == 0 else np.argsort(deltas_1)[::-1][:k_nbs]

        # Store deltas and corresponding neighbor ids
		deltas_0_all.append(np.array(deltas_0)[sort_inds])
		deltas_1_all.append(np.array(deltas_1)[sort_inds])
		nbs_to_remove_all.append(np.array(nb_ids[sort_inds]))

		user_cnt += 1

	deltas_0_flat = np.array(deltas_0_all).flatten()
	deltas_1_flat = np.array(deltas_1_all).flatten()
	nbs_to_remove_flat = np.array(nbs_to_remove_all).flatten()

	user_ids_all = np.repeat(sample_users, k_nbs) # Keep an array with user ids synced with nbs removed

    # Create adjacency matrix
	adj_mat = np.zeros((users_num + items_num, users_num + items_num), dtype=int)
	adj_mat[:users_num, users_num:] = R
	adj_mat[users_num:, :users_num] = R.T

    # TODO: Calculate initial scores without having to calculate absorbing mat only for this
	P = create_recwalk_model_from_adj(adj_mat)
	scores_absorbing = ppr_absorbing_3(P=P,
                                       item_group_ids_0=item_group_mapped_ids_0,
                                       item_group_ids_1=item_group_mapped_ids_1,
                                       users_num=users_num,
                                       gamma=0.15)

    # Calculate initial PPR scores
	ppr_0.append(np.mean(scores_absorbing[user_group_mapped_ids, 0]))
	ppr_1.append(np.mean(scores_absorbing[user_group_mapped_ids, 1]))

	# Remove based on overall sorting and recalculate PPR
	sort_inds = np.argsort(deltas_0_flat)[::-1] if genre_absorbing_col == 0 else np.argsort(deltas_1_flat)[::-1]

	num_rounds = ceil(sample_size*k_nbs/nbs_per_round)
	for r in range(num_rounds):
		if r < (num_rounds - 1):
			inds_to_remove = sort_inds[r*nbs_per_round:(r+1)*nbs_per_round]
		else:
			inds_to_remove = sort_inds[r*nbs_per_round:]

		adj_mat[user_ids_all[inds_to_remove][:, np.newaxis], users_num + nbs_to_remove_flat[inds_to_remove]] = 0

		P = create_recwalk_model_from_adj(adj_mat)
		scores_absorbing = ppr_absorbing_3(P=P,
                                           item_group_ids_0=item_group_mapped_ids_0,
                                           item_group_ids_1=item_group_mapped_ids_1,
                                           users_num=users_num,
                                           gamma=0.15)

		ppr_0.append(np.mean(scores_absorbing[user_group_mapped_ids, 0]))
		ppr_1.append(np.mean(scores_absorbing[user_group_mapped_ids, 1]))

		print(f"[{datetime.datetime.now()}] Round {r + 1}/{num_rounds} completed.")

    # Keep time ended
	end_wall_t = time.time()
	print(f"Wall time: {round((end_wall_t-start_wall_t)/60, 4)} minutes")

    ##### EXPORT RESULTS TO FILE #####
	if export_results:
		with open(f"{export_dir_path}/{export_filename}.tsv", 'w') as f:
    		# Write header lines
			f.write('# File format:\n')
			f.write('# <users_num> <k_nbs> <num_rounds>\n')
			f.write('# <ppr_0_initial> <ppr_0_0> <ppr_0_1> [...] <ppr_0_m>\n')
			f.write('# <ppr_1_initial> <ppr_1_0> <ppr_1_1> [...] <ppr_1_m>\n')
			f.write('# <item_id_0> <item_id_1> [...] <item_id_m>\n')
			f.write('# <user_id_0> <user_id_1> [...] <user_id_m>\n')

			f.write(f"{sample_users.shape[0]}\t{k_nbs}\t{num_rounds}\n")

			for i in range(len(ppr_0)):
				if i != (len(ppr_0) - 1):
					f.write(f"{ppr_0[i]}\t")
				else:
					f.write(f"{ppr_0[i]}\n")

			for i in range(len(ppr_1)):
				if i != (len(ppr_1) - 1):
					f.write(f"{ppr_1[i]}\t")
				else:
					f.write(f"{ppr_1[i]}\n")

			for ind in sort_inds:
				if ind != sort_inds[-1]:
					f.write(f"{item_original_ids[nbs_to_remove_flat[ind]]}\t")
				else:
					f.write(f"{item_original_ids[nbs_to_remove_flat[ind]]}\n")

			for ind in sort_inds:
				if ind != sort_inds[-1]:
					f.write(f"{user_original_ids[user_ids_all[ind]]}\t")
				else:
					f.write(f"{user_original_ids[user_ids_all[ind]]}")


def calculate_best_k_deltas(R,
			                user_original_ids,
			                item_original_ids,
			                user_group_mapped_ids,
			                item_group_mapped_ids_0,
			                item_group_mapped_ids_1,
						    sample_users,
						    k_nbs,
						    genre_absorbing_col,
						    export_dir_path=None,
			                export_filename=None,
			                export_results=True):
	'''TODO: Write description
	'''
	users_num, items_num = R.shape[0], R.shape[1]
	sample_size = sample_users.shape[0]

	deltas_0_all, deltas_1_all = [], []
	nbs_to_remove_all = []

	P = create_recwalk_model(R) # Create the model

	user_cnt = 0
	start_wall_t = time.time() # Keep time started
	for user_id in sample_users:
		nb_ids = np.nonzero(R[user_id])[0] # Find current user neighbors

		print(f"[{datetime.datetime.now()}] User {user_cnt + 1} - id: {user_original_ids[user_id]}:" +
                                            f"#neighbors={nb_ids.shape[0]}")

	    # Calculate scores using ppr_absorbing_4 once
		scores_absorbing = ppr_absorbing_4(P                = P,
										   user_x_id        = user_id,
										   item_group_ids_0 = item_group_mapped_ids_0,
										   item_group_ids_1 = item_group_mapped_ids_1,
										   users_num        = users_num,
										   gamma            = 0.15)

		scores_absorbing_items = scores_absorbing[users_num:(users_num+items_num)]

		d_term_1 = np.mean(scores_absorbing[user_group_mapped_ids, -1])
		deltas_0, deltas_1 = [], []
		for nb_to_remove in nb_ids:
			deltas_0.append(d_term_1*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
		                                              user_x_id              = user_id,
		                                              nb_ids                 = nb_ids,
		                                              nbs_to_remove          = np.array([nb_to_remove]),
		                                              genre_absorbing_col    = 0,
		                                              gamma                  = 0.15))

			deltas_1.append(d_term_1*calculate_lambda(scores_absorbing_items = scores_absorbing_items,
		                                              user_x_id              = user_id,
		                                              nb_ids                 = nb_ids,
		                                              nbs_to_remove          = np.array([nb_to_remove]),
		                                              genre_absorbing_col    = 1,
		                                              gamma                  = 0.15))

        # Decide which category to increase based on genre_absorbing_col value
		sort_inds = np.argsort(deltas_0)[::-1][:k_nbs] if genre_absorbing_col == 0 else np.argsort(deltas_1)[::-1][:k_nbs]

        # Store deltas and corresponding neighbor ids
		deltas_0_all.append(np.array(deltas_0)[sort_inds])
		deltas_1_all.append(np.array(deltas_1)[sort_inds])
		nbs_to_remove_all.append(np.array(nb_ids[sort_inds]))

		user_cnt += 1

	deltas_0_flat = np.array(deltas_0_all).flatten()
	deltas_1_flat = np.array(deltas_1_all).flatten()
	nbs_to_remove_flat = np.array(nbs_to_remove_all).flatten()

	user_ids_all = np.repeat(sample_users, k_nbs) # Keep an array with user ids synced with nbs removed

	# Apply overall sorting
	sort_inds = np.argsort(deltas_0_flat)[::-1] if genre_absorbing_col == 0 else np.argsort(deltas_1_flat)[::-1]

    # Keep time ended
	end_wall_t = time.time()
	print(f"Wall time: {round((end_wall_t-start_wall_t)/60, 4)} minutes")

    ##### EXPORT RESULTS TO FILE #####
	if export_results:
		with open(f"{export_dir_path}/{export_filename}.tsv", 'w') as f:
    		# Write header lines
			f.write('# File format:\n')
			f.write('# <users_num> <k_nbs>\n')
			f.write('# <delta_0_0> <delta_0_1> [...] <delta_0_m>\n')
			f.write('# <delta_1_0> <delta_1_1> [...] <delta_1_m>\n')
			f.write('# <item_id_0> <item_id_1> [...] <item_id_m>\n')
			f.write('# <user_id_0> <user_id_1> [...] <user_id_m>\n')

			f.write(f"{sample_users.shape[0]}\t{k_nbs}\n")

			for ind in sort_inds:
				if ind != sort_inds[-1]:
					f.write(f"{deltas_0_flat[ind]}\t")
				else:
					f.write(f"{deltas_0_flat[ind]}\n")

			for ind in sort_inds:
				if ind != sort_inds[-1]:
					f.write(f"{deltas_1_flat[ind]}\t")
				else:
					f.write(f"{deltas_1_flat[ind]}\n")

			for ind in sort_inds:
				if ind != sort_inds[-1]:
					f.write(f"{item_original_ids[nbs_to_remove_flat[ind]]}\t")
				else:
					f.write(f"{item_original_ids[nbs_to_remove_flat[ind]]}\n")

			for ind in sort_inds:
				if ind != sort_inds[-1]:
					f.write(f"{user_original_ids[user_ids_all[ind]]}\t")
				else:
					f.write(f"{user_original_ids[user_ids_all[ind]]}")