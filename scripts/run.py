# Useful starting lines
from implementations import *

import numpy as np
import sys


### LOAD DATASET ###
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


### SEPARATE AND CLEAN DATASETS ###
tX_list, ids_list, y_list = separate_dataset(tX, ids, y)
tX_test_list, ids_test_list = separate_dataset(tX_test, ids_test) 

tX_list, tX_test_list, y_list = clean_data(tX_list, tX_test_list, y_list)

np.random.seed(2)

### CHOICE OF METHOD ###
# FUNCTION
# 1 = least squares GD
# 2 = least squares SGD
# 3 = least squares
# 4 = ridge regression
# 5 = logistic regression
# 6 = reg logistic regression
function = int(sys.argv[1]) if len(sys.argv) == 2 else 3

### TRAINING ###
weights_list = []
loss_list = []
mat_tX_test_list = []

for i in range(6):
    ###########
    # For each method, we create the degree, lambda and gamma vectors that allowed us to get the best score
    # For GD methods, we set the initial weights and max number of iterations
    # We perform polynomial expansion using build_poly_log with the best degree
    # We train the model (using the best hyper parameters), this gives us the weights that will be used for inference
    ###########

    if (function == 1):
        degree_vec = [2, 2, 1, 2, 2, 2]
        gamma_vec = [0.046415888336127774, 0.046415888336127774, 0.046415888336127774, 0.046415888336127774, 0.046415888336127774, 0.046415888336127774]
        max_iters = 300
        
        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        initial_w = np.zeros(mat_tX.shape[1])

        w, l = least_squares_GD(y_list[i], mat_tX, initial_w, max_iters, gamma_vec[i])

    elif (function == 2):
        degree_vec = [2, 1, 2, 2, 1, 1]
        gamma_vec = [0.0001, 0.01, 0.0001, 0.001, 0.01, 0.01]
        max_iters = 1000

        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        initial_w = np.zeros(mat_tX.shape[1])
        
        w, l = least_squares_SGD(y_list[i], mat_tX, initial_w, max_iters, gamma_vec[i])

    elif (function == 3):
        degree_vec = [10, 13, 6, 14, 5, 14]

        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        w, l = least_squares(y_list[i], mat_tX)

    elif (function == 4):
        degree_vec = [11, 15, 7, 14, 8, 13]
        lambda_vec = [7.196856730011529e-08, 1e-10, 1e-10, 0.001389495494373139, 0.0002682695795279727, 1e-10]
        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        w, l = ridge_regression(y_list[i], mat_tX, lambda_vec[i])
    
    elif (function == 5):
        degree_vec = [2,3,2,3,3,3]
        gamma_vec = [0.1, 0.1, 0.1, 0.1, 0.01, 0.1]
        max_iter= 7000

        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        initial_w = np.zeros((mat_tX.shape[1],1))

        w, l = logistic_regression(y_list[i], mat_tX, initial_w, max_iters, gamma_vec[i])

    elif (function == 6):
        degree_vec = [2, 3, 2, 3, 3, 3]
        lambda_vec = [1e-08, 3.1622776601683795e-05, 3.1622776601683795e-05, 1e-08, 1e-08, 1e-08]
        gamma_vec = [0.1, 0.1, 0.1, 0.1, 0.01, 0.1]
        max_iters = 7000

        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        initial_w = np.zeros((mat_tX.shape[1],1))

        w, l = reg_logistic_regression(y_list[i], mat_tX, initial_w, max_iters, lambda_vec[i], gamma_vec[i])

    else:
        print('error function name')

    weights_list.append(w)
    loss_list.append(l)
    mat_tX_test_list.append(mat_tX_test)    


### INFERENCE ###
y_pred_list = separated_eval(weights_list, mat_tX_test_list) 


### CONCATENATION OF THE RESULTS AND SUBMISSION FILE CREATION ###
y_pred = np.concatenate((y_pred_list[0], y_pred_list[1], y_pred_list[2], y_pred_list[3], y_pred_list[4], y_pred_list[5]))
ids_test_sub = np.concatenate((ids_test_list[0], ids_test_list[1], ids_test_list[2], ids_test_list[3], ids_test_list[4], ids_test_list[5]))

OUTPUT_PATH = 'result.csv'
create_csv_submission(ids_test_sub, y_pred, OUTPUT_PATH)
 
