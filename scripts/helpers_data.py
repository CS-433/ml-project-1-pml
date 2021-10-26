import numpy as np
from implementations import *

def separate_dataset(tX, ids, y = None, logistic = False):
    tX_list = []
    y_list = []
    ids_list = []
    for i in range(6):
        if i < 4:            
            indices = np.logical_and((np.isclose(tX[:,22], np.floor((i+0.01) / 2.0))), (tX[:,0] == -999 if i % 2 == 0 else tX[:,0] != -999))
        else:
            indices = np.logical_and((np.any((np.isclose(tX[:,22], np.floor((i+0.01) / 2.0)), np.isclose(tX[:,22], np.floor((i+0.01) / 2.0)+1)), axis = 0)), (tX[:,0] == -999 if i % 2 == 0 else tX[:,0] != -999))
        tX_list.append(tX[indices])
        ids_list.append(ids[indices])

        tX_list[i] = np.delete(tX_list[i], 22, axis=1) #Delete 22nd column
        tX_list[i] = tX_list[i][:, ~np.all(tX_list[i][1:] == tX_list[i][:-1], axis=0)] #Delete column with all the same values (so the columns of -999)

        # mean = np.mean(tX_list[i], axis = 0, where = tX_list[i] != -999)
        # tX_with_NaN=np.where(tX_list[i] == -999, np.nan, tX_list[i])
        # median = np.nanmedian(tX_with_NaN, axis = 0)

        # tX_list[i] = np.where(tX_list[i] == -999, median, tX_list[i])

        tX_list[i] = normalize(tX_list[i])
        #tX_list[i] = standardize(tX_list[i])


        if y is not None:
            y_l = y[indices]
            if logistic:
                y_list.append(np.where(y_l == -1, 0, y_l))
            else:
                y_list.append(y_l)
    if y is not None:
        return tX_list, ids_list, y_list
    return tX_list, ids_list


def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    #améliorer pour ne pas sauter qq samples ?
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    #np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def build_poly_log(x_tr, degree, log = False, x_te = None):
    poly_tr = np.ones((len(x_tr), 1))
    for deg in range(1, degree+1):
        poly_tr = np.c_[poly_tr, np.power(x_tr, deg)]

    if x_te is not None:
        poly_te = np.ones((len(x_te), 1))
        for deg in range(1, degree+1):
            poly_te = np.c_[poly_te, np.power(x_te, deg)]

    log_tr = np.ones((len(x_tr), 1))
    log_te = np.ones((len(x_te), 1))

    if log:
        for i in range(x_tr.shape[1]):
            if (np.all(x_tr[:,i] > 0) and np.all(x_te[:,i] > 0)):
                log_tr = np.c_[log_tr, np.log(x_tr[:,i])]
                log_te = np.c_[log_te, np.log(x_te[:,i])]
        poly_tr = np.c_[poly_tr, log_tr[:,1:]]
        poly_te = np.c_[poly_te, log_te[:,1:]]
    return poly_tr, poly_te


def cross_validation(y, x, k_indices, k, degree, function, args = None, log = False):
    """return the loss of ridge regression."""

    indices_te = k_indices[k]
    indices_tr = np.delete(k_indices, k, axis=0)
    indices_tr = np.concatenate(indices_tr, axis= None)
    x_tr = x[indices_tr]
    y_tr = y[indices_tr]
    x_te = x[indices_te]
    y_te = y[indices_te]
    
    x_tr_poly, x_te_poly = build_poly_log(x_tr, degree, log, x_te)
    
    if (function == ridge_regression):
        weights, loss_tr = ridge_regression(y_tr, x_tr_poly, args[0])
    elif (function == least_squares):
        weights, loss_tr = least_squares(y_tr, x_tr_poly)

    loss_te = compute_loss(y_te, x_te_poly, weights)
    
    return loss_tr, loss_te, weights

def cross_validation_log_len(y, x, k_indices, k, degree, lambda_ , gamma , log = False):
    """return the loss of ridge regression."""

    max_iter= 100
    

    indices_te = k_indices[k]
    indices_tr = np.delete(k_indices, k, axis=0)
    indices_tr = np.concatenate(indices_tr, axis= None)
    x_tr = x[indices_tr]
    y_tr = y[indices_tr]
    x_te = x[indices_te]
    y_te = y[indices_te]
    
    x_tr_poly, x_te_poly = build_poly_log(x_tr, degree, log, x_te)
    initial_w = np.zeros((x_tr_poly.shape[1], 1))

    loss_tr, weights = logistic_regression_penalized_gradient_descent(y_tr, x_tr_poly, initial_w, max_iter, gamma, lambda_)
    
    loss_te = compute_loss_log(y_te, x_te_poly, weights)
    
    return loss_tr, loss_te, weights


def separated_eval(weights_list, tX_test_list, logistic = False):
    y_pred_list = []
    for i in range(6):
        if logistic:
            y_pred_list.append(predict_labels_log(weights_list[i], tX_test_list[i]))
        else:
            y_pred_list.append(predict_labels(weights_list[i], tX_test_list[i]))
    return y_pred_list
