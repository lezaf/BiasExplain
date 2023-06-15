'''The file contains functions for synthetic data generation.

'''
import os

import numpy as np
import random

from   scipy.special import zeta

def generate_synthetic_graph(num_users,
                             num_items,
                             r_u,
                             r_i,
                             p00,
                             p11,
                             ratings_per_user,
                             zipf_a_0=1,
                             zipf_a_1=1,
                             k=1,
                             return_adj_mat=False,
                             export_to_file=False,
                             path=None,
                             filename=None):
    '''Generates a synthetic bipartite graph of users-items. Popularities in items
    can be determined using Zipf's law.
       
    Notes:
        The popularities are are generated with following equation:
            
                        p(t) = t^(-zipf_a)/zeta(zipf_a)
        
        where:
            t: Integers 1...num_items, with 1 to be the most popular item
            zipf_a: Zipf 'a' parameter
            zeta: Riemann zeta function
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        r_u (float): Percentage of users of group U_0
        r_i (float): Percentage of items of group I_0
        p00 (float): Probability a user in U_0 to rate an item in I_0 (or P(I_0 | U_0))
        p11 (float): Probability a user in U_1 to rate an item in I_1 (or P(I_1 | U_1))
        ratings_per_user (int): Number of ratings each user will make
        zipf_a_0 (float): Parameter 'a' for the probability density of Zipf's distribution in I_0
            Default: 1 (uniform)
        zipf_a_1 (float): Parameter 'a' for the probability density of Zipf's distribution in I_1
            Default: 1 (uniform)
        k (int): Number of initial ratings for each item
    Returns:
        R (numpy.2darray): The 0/1 interaction array
    '''
    # Arguments check
    if r_u < 0 or r_u > 1:
        raise ValueError('r_u should be in the range [0,1].')
    if r_i < 0 or r_i > 1:
        raise ValueError('r_i should be in the range [0,1].')
    if p00 < 0 or p00 > 1:
        raise ValueError('p00 should be in the range [0,1].')
    if p11 < 0 or p11 > 1:
        raise ValueError('p11 should be in the range [0,1].')
    if ratings_per_user > num_items:
        raise ValueError('ratings_per_user cannot exceed num_items.')
    if zipf_a_0 < 1 :
        raise ValueError('zipf_a_0 should be greater than or equal to 1.\nDefault: 1 (uniform)')
    if zipf_a_1 < 1:
        raise ValueError('zipf_a_1 should be greater than or equal to 1.\nDefault: 1 (uniform)')
    if k < 0 or k > ratings_per_user:
        raise ValueError('k should be between 0 and ratings_per_user.')
    
    # Create the nxm interaction matrix
    R = np.zeros((num_users, num_items), dtype=int)
    
    # Calculate number of users/items per groups
    num_users_0, num_users_1 = round(r_u*num_users), num_users - round(r_u*num_users)
    num_items_0, num_items_1 = round(r_i*num_items), num_items - round(r_i*num_items)

    if zipf_a_0 == 1 and zipf_a_1 == 1: # Uniform popularity
        w_0, w_1 = np.repeat(1/num_items_0, num_items_0), np.repeat(1/num_items_1, num_items_1)
    elif zipf_a_0 == 1 and zipf_a_1 > 1:
        w_0 = np.repeat(1/num_items_0, num_items_0)

        w_1 = np.power(np.arange(1, num_items_1 + 1), -zipf_a_1)/zeta(zipf_a_1)
        w_1 = w_1/np.sum(w_1) # Normalize
    elif zipf_a_0 > 1 and zipf_a_1 == 1:
        w_1 = np.repeat(1/num_items_1, num_items_1)

        w_0 = np.power(np.arange(1, num_items_0 + 1), -zipf_a_0)/zeta(zipf_a_0)
        w_0 = w_0/np.sum(w_0)
    else:
        w_0 = np.power(np.arange(1, num_items_0 + 1), -zipf_a_0)/zeta(zipf_a_0)
        w_0 = w_0/np.sum(w_0)
        w_1 = np.power(np.arange(1, num_items_1 + 1), -zipf_a_1)/zeta(zipf_a_1)
        w_1 = w_1/np.sum(w_1)
    
    u_ratings_cnt, i_ratings_cnt = np.zeros(num_users, dtype=int), np.zeros(num_items, dtype=int)

    p_U_0_from_I_0 = p00 / (p00 + (1 - p11)) # p(U_0 | I_0)
    p_U_1_from_I_1 = p11 / (p11 + (1 - p00)) # p(U_1 | I_1)

    # Allocate initial ratings uniformly
    U_0_p = np.concatenate((np.repeat(p_U_0_from_I_0, num_items_0), np.repeat(1 - p_U_1_from_I_1, num_items_1)))
    for i in range(num_items):
        cur_U_0_p = U_0_p[i]
        groups_p = np.random.random(size=k) # Get random sequence
        
        U_0_r_cnt = np.count_nonzero(groups_p < cur_U_0_p)
        U_1_r_cnt = k - U_0_r_cnt
        
        U_0_available_inds = np.nonzero(u_ratings_cnt[:num_users_0] < ratings_per_user)[0]
        U_1_available_inds = np.nonzero(u_ratings_cnt[num_users_0:] < ratings_per_user)[0]
        if U_0_available_inds.shape[0] < U_0_r_cnt:
            U_0_inds = U_0_available_inds
            U_1_inds = np.random.choice(U_1_available_inds, size=U_1_r_cnt + U_0_r_cnt - U_0_available_inds.shape[0], replace=False)
        elif U_1_available_inds.shape[0] < U_1_r_cnt:
            U_1_inds = U_1_available_inds
            U_0_inds = np.random.choice(U_0_available_inds, size=U_0_r_cnt + U_1_r_cnt - U_1_available_inds.shape[0], replace=False)
        else:
            U_0_inds = np.random.choice(U_0_available_inds, size=U_0_r_cnt, replace=False)
            U_1_inds = np.random.choice(U_1_available_inds, size=U_1_r_cnt, replace=False)

        R[U_0_inds, i] = 1
        R[U_1_inds+num_users_0, i] = 1
        
        u_ratings_cnt[U_0_inds] += 1
        u_ratings_cnt[U_1_inds + num_users_0] += 1
        
        i_ratings_cnt[i] += k
        
    del U_0_p # Free unnecessary memory

    # Allocate ratings with weights
    I_0_p = np.concatenate((np.repeat(p00, num_users_0), np.repeat(1 - p11, num_users_1)))
    for i in range(num_users):
        if u_ratings_cnt[i] == ratings_per_user: continue
        cur_I_0_p = I_0_p[i]
        
        remaining_r = ratings_per_user - u_ratings_cnt[i] # Remaining #ratings user has
        groups_p = np.random.random(size=remaining_r) # Get random sequence
        
        I_0_r_cnt = np.count_nonzero(groups_p < cur_I_0_p)
        I_1_r_cnt = remaining_r - I_0_r_cnt
        
        I_0_available_inds = np.where(R[i, :num_items_0] == 0)[0]
        I_1_available_inds = np.where(R[i, num_items_0:] == 0)[0]
        if I_0_available_inds.shape[0] < I_0_r_cnt:
            I_0_inds = I_0_available_inds
            I_1_inds = np.random.choice(I_1_available_inds,
                                        size=I_1_r_cnt + I_0_r_cnt - I_0_available_inds.shape[0],
                                        replace=False,
                                        p=w_1[I_1_available_inds]/np.sum(w_1[I_1_available_inds]))
        elif U_1_available_inds.shape[0] < I_1_r_cnt:
            I_1_inds = I_1_available_inds
            I_0_inds = np.random.choice(I_0_available_inds,
                                        size=I_0_r_cnt + I_1_r_cnt - I_1_available_inds.shape[0],
                                        replace=False,
                                        p=w_0[I_0_available_inds]/np.sum(w_0[I_0_available_inds]))
        else:
            I_0_inds = np.random.choice(I_0_available_inds, size=I_0_r_cnt, replace=False,
                                        p=w_0[I_0_available_inds]/np.sum(w_0[I_0_available_inds]))
            I_1_inds = np.random.choice(I_1_available_inds, size=I_1_r_cnt, replace=False,
                                        p=w_1[I_1_available_inds]/np.sum(w_1[I_1_available_inds]))
        
        R[i, I_0_inds] = 1
        R[i, I_1_inds + num_items_0] = 1
        
        u_ratings_cnt[i] += remaining_r
        
        i_ratings_cnt[I_0_inds] += 1
        i_ratings_cnt[I_1_inds + num_items_0] += 1

    if export_to_file:
        # Write info file
        with open(f"{os.path.join(path, filename)}.info", 'w') as f:
            f.write(f"num_users\t{num_users}\n")
            f.write(f"num_items\t{num_items}\n")
            f.write(f"r_u\t{r_u}\n")
            f.write(f"r_i\t{r_i}\n")
            f.write(f"p00\t{p00}\n")
            f.write(f"p11\t{p11}\n")
            f.write(f"ratings_per_user\t{ratings_per_user}\n")
            f.write(f"zipf_a_0\t{zipf_a_0}\n")
            f.write(f"zipf_a_1\t{zipf_a_1}")

        # Write edge list
        with open(f"{os.path.join(path, filename)}.edges", 'w') as f:
            for i in range(num_users):
                rating_inds = np.nonzero(R[i])[0]
                for ind in rating_inds:
                    f.write(f"{i}\t{ind}\n")

    if return_adj_mat:
        return R


def read_graph(dir_path,
               filename):
    
    with open(f"{os.path.join(dir_path, filename)}.info") as f:
        info_dict = dict()

        for line in f:
            info_dict[line.split()[0]] = line.split()[0]
            if line.split()[1].isdigit():
                info_dict[line.split()[0]] = int(line.split()[1])
            else:
                info_dict[line.split()[0]] = float(line.split()[1])

    with open(f"{os.path.join(dir_path, filename)}.edges") as f:
        R = np.zeros((info_dict['num_users'], info_dict['num_items']), dtype=int)

        for line in f:
            u_id, i_id = int(line.split()[0]), int(line.split()[1])
            R[u_id, i_id] = 1

    return info_dict, R