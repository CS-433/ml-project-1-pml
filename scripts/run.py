# Useful starting lines
import numpy as np
from implementations import *


### LOAD DATASET ###
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


### SEPARATE AND CLEAN DATASETS ###
tX_list, ids_list, y_list = separate_dataset(tX, ids, y)
tX_test_list, ids_test_list = separate_dataset(tX_test, ids_test) 

tX_list, tX_test_list, y_list = clean_data(tX_list, tX_test_list, y_list)


### CHOICE OF METHOD ###
# FUNCTION
# 1 = least squares GD
# 2 = least squares SGD
# 3 = least squares
# 4 = ridge regression
# 5 = logistic regression
# 6 = reg logistic regression
function = 4


### INFERENCE ###
weights_list = []
loss_list = []
mat_tX_test_list = []

for i in range(6):

    if (function == 1):
        degree_vec = [2, 2, 1, 2, 2, 2]
        gamma_vec = [0.046415888336127774, 0.046415888336127774, 0.046415888336127774, 0.046415888336127774, 0.046415888336127774, 0.046415888336127774]
        max_iters = 300
        initial_w = np.zeros(mat_tX.shape[1])

        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        w, l = least_squares_GD(y_list[i], mat_tX, initial_w, max_iters, gamma_vec[i])

    elif (function == 2):
        degree_vec = [2, 1, 2, 2, 1, 1]
        gamma_vec = [0.0001, 0.01, 0.0001, 0.001, 0.01, 0.01]
        max_iters = 1000
        initial_w = np.zeros(mat_tX.shape[1])
        
        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
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

    elif (function == 6):
        degree_vec = [2, 3, 2, 4, 2, 3]
        lambda_vec = [1e-08, 5.62341325190349e-07, 1e-08, 3.1622776601683795e-05, 0.0017782794100389228, 5.62341325190349e-07]
        gamma_vec = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        max_iters = 3000
        initial_w = np.zeros((mat_tX.shape[1],1))

        mat_tX, mat_tX_test = build_poly_log(tX_list[i], degree_vec[i], tX_test_list[i], i)
        w, l = reg_logistic_regression(y_list[i], mat_tX, initial_w, max_iters, lambda_vec[i], gamma_vec[i])

    else:
        print('error function name')

    weights_list.append(w)
    loss_list.append(l)
    mat_tX_test_list.append(mat_tX_test)    


### MODEL EVALUATION ###s

y_pred_list = separated_eval(weights_list, mat_tX_test_list) 

y_pred = np.concatenate((y_pred_list[0], y_pred_list[1], y_pred_list[2], y_pred_list[3], y_pred_list[4], y_pred_list[5]))
ids_test_sub = np.concatenate((ids_test_list[0], ids_test_list[1], ids_test_list[2], ids_test_list[3], ids_test_list[4], ids_test_list[5]))

OUTPUT_PATH = 'result.csv'
create_csv_submission(ids_test_sub, y_pred, OUTPUT_PATH)
 