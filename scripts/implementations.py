import numpy as np
from proj1_helpers import *

def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def standardize(data):
    return (data - np.average(data)) / (np.std(data))

def compute_loss(y, tx, w):
    """Calculate the loss."""
    #loss = ((y - tx.dot(w))**2).sum()/(2*len(y))   #MSE
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    # loss = np.absolute(y - tx.dot(w)).sum()/ len(y)   #MAE
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    grad = -tx.T.dot(y - tx.dot(w))/len(y)
    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - (gamma * gradient)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss

def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - (gamma * gradient)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = (((y - tx@w)**2).sum()) / (2 * len(y))
    return w, mse 

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_pr = lambda_ * 2 * len(y)
    w = np.linalg.solve(tx.T @ tx + lambda_pr * np.eye(tx.shape[1]), tx.T @ y)
    loss = ((y - tx @ w)**2).sum() * 0.5 / len(y)
    return w, loss

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    for deg in range(degree):
        matdeg=np.full((x.shape[0], x.shape[1]), deg)
        x=np.c_[x, x**matdeg]
    return x

def build_log(x):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x=np.c_[x, np.log(x)]
    x=np.nan_to_num(x, nan=-999)
    return x

def build_poly_separated(x, degree):
    mat_tX = []
    for i in range(4):
        mat_tX.append(build_poly(x[i], degree))
    return mat_tX

def build_log_separated(x):
    mat_tX = []
    for i in range(4):
        mat_tX.append(build_poly(x[i]))
    return mat_tX

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    length = len(y)
    indices = np.random.permutation(len(y))#np.arange(length)
    #np.random.shuffle(indices)
    length_tr = np.ceil(ratio * length).astype(int)
    x_train = x[indices][:length_tr]
    x_test = x[indices][length_tr:]
    y_train = y[indices][:length_tr]
    y_test = y[indices][length_tr:]
    return x_train, x_test, y_train, y_test

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    indices_te=k_indices[k]
    indices_tr=np.delete(k_indices, k, axis=0)
    indices_tr= np.concatenate(indices_tr, axis= None)
    x_tr=x[indices_tr]
    y_tr=y[indices_tr]
    x_te=x[indices_te]
    y_te=y[indices_te]
    
    
    x_tr_poly=build_poly(x_tr, degree)
    x_te_poly=build_poly(x_te, degree)
    
    weights, _= ridge_regression(y_tr, x_tr_poly, lambda_)
    
    loss_tr=compute_loss(y_tr, x_tr_poly, weights)
    loss_te=compute_loss(y_te, x_te_poly, weights)
    
    return loss_tr, loss_te, weights



def separate_dataset(tX, ids, y = None):
    tX_list = []
    y_list = []
    ids_list = []
    for i in range(4):
        indices = np.isclose(tX[:,22], i)
        tX_list.append(tX[indices])
        ids_list.append(ids[indices])
        if y is not None:
            y_list.append(y[indices])
    if y is not None:
        return tX_list, ids_list, y_list
    return tX_list, ids_list

def separated_train(tX_list, y_list, function, args):
    weights = []
    loss = []
    for i in range(4):
        w, l = function(y_list[i], tX_list[i], args)
        weights.append(w)
        loss.append(l)
    return weights, loss

def separated_eval(weights_list, tX_test_list):
    y_pred_list = []
    for i in range (4):
        y_pred_list.append(predict_labels(weights_list[i], tX_test_list[i]))
    return y_pred_list

