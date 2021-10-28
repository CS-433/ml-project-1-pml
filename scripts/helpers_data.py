import numpy as np
from proj1_helpers import predict_labels

#####################
def normalize(data):
    return (data-np.min(data, axis = 0))/(np.max(data, axis = 0)-np.min(data, axis = 0))

def standardize(data, m = None, s = None):
    if m is None:
        m = np.average(data, axis = 0)
        s = np.std(data, axis = 0)
        data = (data - m) / s
        return data, m , s
    else:
        data = (data - m) / s
        return data

def compute_score(y, x, weights):
    y_pred = predict_labels(weights, x)
    score = (y == y_pred).sum()
    return score


def clean_data(tX_list, tX_test_list, y_list):

    alpha = 0.98
    for i in range(6):
        dist_fr_mean = np.std(tX_list[i], axis = 0)/(np.sqrt(1-alpha))
        mean = np.average(tX_list[i], axis = 0)

        tX_list[i] = np.where(np.abs(tX_list[i]-mean) > dist_fr_mean, np.nan,tX_list[i])
        y_list[i] = y_list[i][~np.isnan(tX_list[i]).any(axis=1)]
        tX_list[i] = tX_list[i][~np.isnan(tX_list[i]).any(axis=1)]
        

    tX_list[0]=np.delete(tX_list[0], [2,4,7,10,13,15], axis=1)
    tX_list[1]=np.delete(tX_list[1], [3,5,11,14,16,17], axis=1)
    tX_list[2]=np.delete(tX_list[2], [7,9,10,12,13,15,16,17,19], axis=1)
    tX_list[3]=np.delete(tX_list[3], [3,5,6,11,12,14,15,16,17,18,20,21], axis=1)
    tX_list[4]=np.delete(tX_list[4], [10,15,16,17,18,19,23,24,26], axis=1)
    tX_list[5]=np.delete(tX_list[5], [3,8,9,10,15,16,18,19,20,21,22,24,25,27,28], axis=1)

    tX_test_list[0]=np.delete(tX_test_list[0], [2,4,7,10,13,15], axis=1)
    tX_test_list[1]=np.delete(tX_test_list[1], [3,5,11,14,16,17], axis=1)
    tX_test_list[2]=np.delete(tX_test_list[2], [7,9,10,12,13,15,16,17,19], axis=1)
    tX_test_list[3]=np.delete(tX_test_list[3], [3,5,6,11,12,14,15,16,17,18,20,21], axis=1)
    tX_test_list[4]=np.delete(tX_test_list[4], [10,15,16,17,18,19,23,24,26], axis=1)
    tX_test_list[5]=np.delete(tX_test_list[5], [3,8,9,10,15,16,18,19,20,21,22,24,25,27,28], axis=1)

    tX_list[0] = np.append(tX_list[0], np.log(tX_list[0][:,[0,1,3,4,5,7,9,10]]), axis = 1)
    tX_list[1] = np.append(tX_list[1], np.log(tX_list[1][:,[0,2,4,7,9,11]]), axis = 1)
    tX_list[2] = np.append(tX_list[2], np.log(tX_list[2][:,[0,1,4,5,6,7,8,9,11]]), axis = 1)
    tX_list[3] = np.append(tX_list[3], np.log(tX_list[3][:,[0,2,4,6]]), axis = 1)
    tX_list[4] = np.append(tX_list[4], np.log(tX_list[4][:,[0,1,2,4,7,8,9,11,14,15,18]]), axis = 1)
    tX_list[5] = np.append(tX_list[5], np.log(tX_list[5][:,[0,2,4,9]]), axis = 1)

    tX_test_list[0] = np.append(tX_test_list[0], np.log(tX_test_list[0][:,[0,1,3,4,5,7,9,10]]), axis = 1)
    tX_test_list[1] = np.append(tX_test_list[1], np.log(tX_test_list[1][:,[0,2,4,7,9,11]]), axis = 1)
    tX_test_list[2] = np.append(tX_test_list[2], np.log(tX_test_list[2][:,[0,1,4,5,6,7,8,9,11]]), axis = 1)
    tX_test_list[3] = np.append(tX_test_list[3], np.log(tX_test_list[3][:,[0,2,4,6]]), axis = 1)
    tX_test_list[4] = np.append(tX_test_list[4], np.log(tX_test_list[4][:,[0,1,2,4,7,8,9,11,14,15,18]]), axis = 1)
    tX_test_list[5] = np.append(tX_test_list[5], np.log(tX_test_list[5][:,[0,2,4,9]]), axis = 1)

    for i in range(6):
        tX_list[i], mean, std = standardize(tX_list[i])
        tX_test_list[i] = standardize(tX_test_list[i], mean, std)

    return tX_list, tX_test_list, y_list
#####################

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

        #tX_list[i] = normalize(tX_list[i])
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


def build_poly_log(x_tr, degree, x_te = None, dataset = 0):
    num = 0
    if dataset == 0:
        num = 11
    elif dataset == 1:
        num = 12
    elif dataset == 2:
        num = 12
    elif dataset == 3:
        num = 10
    elif dataset == 4:
        num = 19
    else:
        num = 14

    poly_tr = np.ones((len(x_tr), 1))
    for deg in range(1, degree+1):
        poly_tr = np.c_[poly_tr, np.power(x_tr[:,:num], deg)]

    if x_te is not None:
        poly_te = np.ones((len(x_te), 1))
        for deg in range(1, degree+1):
            poly_te = np.c_[poly_te, np.power(x_te[:,:num], deg)]

    # log_tr = np.ones((len(x_tr), 1))
    # log_te = np.ones((len(x_te), 1))

    # if log:
    #     for i in range(x_tr.shape[1]):
    #         if (np.all(x_tr[:,i] > 0) and np.all(x_te[:,i] > 0)):
    #             log_tr = np.c_[log_tr, np.log(x_tr[:,i])]
    #             log_te = np.c_[log_te, np.log(x_te[:,i])]
    #     poly_tr = np.c_[poly_tr, log_tr[:,1:]]
    #     poly_te = np.c_[poly_te, log_te[:,1:]]

    # poly_tr[:,1:] = standardize(poly_tr[:,1:])
    # poly_te[:,1:] = standardize(poly_te[:,1:])
    
    return poly_tr, poly_te


