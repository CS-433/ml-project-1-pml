import numpy as np
from proj1_helpers import *

def normalize(data):
    return (data-np.min(data, axis = 0))/(np.max(data, axis = 0)-np.min(data, axis = 0))

def standardize(data):
    return (data - np.average(data, axis = 0)) / (np.std(data, axis = 0))

def sigmoid(t):
    """apply the sigmoid function on t."""
    sigm = 1 / (1 + np.exp(-t))
    return sigm

def compute_loss(y, tx, w):
    """Calculate the loss."""
    loss = ((y - tx.dot(w))**2).sum()/(2*len(y))   #MSE
    # e = y - tx.dot(w)
    # mse = e.dot(e) / (2 * len(e))
    # loss = np.absolute(y - tx.dot(w)).sum()/ len(y)   #MAE
    return loss

def cross_entropy_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- np.sum(loss))
    #loss = -np.sum(y.reshape((-1,1)) * np.log(sigmoid(tx @ w)) + (1-y.reshape((-1,1))) * np.log(1 - sigmoid(tx @ w)))
    #return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    grad = -tx.T.dot(y - tx.dot(w))/len(y)
    return grad

def cross_entropy_gradient(y, tx, w):
    """compute the gradient of cross entropy loss."""
    grad = tx.T @ (sigmoid(tx @ w) - y.reshape((-1,1)))

    #pred = sigmoid(tx.dot(w))
    #grad = tx.T.dot(pred - y)
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
    # mse = (((y - tx@w)**2).sum()) / (2 * len(y))
    mse = compute_loss(y, tx, w)
    return w, mse 

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_pr = lambda_ * 2 * len(y)
    if np.linalg.det(tx.T @ tx + lambda_pr * np.eye(tx.shape[1])) == 0:
        w= np.zeros((tx.shape[1], 1))
        loss= 1000
    else:
        w = np.linalg.solve(tx.T @ tx + lambda_pr * np.eye(tx.shape[1]), tx.T @ y)
    # loss = ((y - tx @ w)**2).sum() * 0.5 / len(y)
        loss = compute_loss(y, tx, w)
    return w, loss

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = cross_entropy_loss(y, tx, w)
    grad = cross_entropy_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    w = initial_w
    loss_prev = 0

    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if iter > 1 and np.abs(loss - loss_prev) < threshold:
            break  
        loss_prev = loss

    print("loss={l}".format(l=cross_entropy_loss(y, tx, w)))
    return w, loss  


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = cross_entropy_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = cross_entropy_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w


def logistic_regression_penalized_gradient_descent(y, tx, initial_w, max_iter, gamma, lambda_):
    # init parameters
    threshold = 1e-8
    losses_prev = 0

    # build tx
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        #if iter % 100 == 0:
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        if iter > 1 and np.abs(loss - loss_prev) < threshold:
            break
        loss_prev = loss
    return loss, w


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

def build_poly_old(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    for deg in range(degree+1):
        matdeg=np.full((x.shape[0], x.shape[1]), deg)
        x=np.c_[x, x**deg]
    return x

def build_poly(x, degree, log = False):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]

    log_ = np.ones((len(x), 1))
    if log:
        for i in range(x.shape[1]):
            if np.all(x[:,i] > 0):
                log_ = np.c_[log_, np.log(x[:,i])]

        poly = np.c_[poly, log_[:,1:]]
    return poly

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


def build_poly_separated(x, degree, log=False):
    mat_tX = []
    for i in range(3):
        mat_tX.append(build_poly(x[i], degree, log))
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

    max_iter= 700
    

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
    

    loss_te = cross_entropy_loss(y_te, x_te_poly, weights)
    
    return loss_tr, loss_te, weights


def grid_search(y, tX, function, log = False, k_fold = 4, degrees = range(1, 15), lambdas = np.logspace(-8, -1, 35)):
    # Ridge regression with K-fold
    k_indices = build_k_indices(y, k_fold)

    rmse_te_tmp = []
    BestLambdaForDeg=[]
    for index_degree, degree in enumerate(degrees):
        rmse_te_tmp2 = []
        for index_lambda, lambda_ in enumerate(lambdas):
            loss_te_tmp = 0
            for k in range(k_fold):
                _, loss_te, _ = cross_validation(y, tX, k_indices, k, degree, function, (lambda_,), log)
                loss_te_tmp = loss_te_tmp + loss_te
            rmse_te_tmp2.append(np.sqrt(2 * loss_te_tmp / k_fold))
        rmse_te_tmp.append(min(rmse_te_tmp2))
        BestLambdaForDeg.append(lambdas[np.argmin(rmse_te_tmp2)])
    BestDeg = degrees[np.argmin(rmse_te_tmp)]
    BestLambda = BestLambdaForDeg[np.argmin(rmse_te_tmp)]
    rmse_te = min(rmse_te_tmp)

    return rmse_te, BestDeg, BestLambda


def grid_search_for_log_reg(y, tX, log = False, k_fold = 4, degrees = range(1, 15), lambdas = np.logspace(-7, -1, 25), gammas = np.logspace(-11, -8, 25)):

    k_indices = build_k_indices(y, k_fold)

    rmse_te_tmp = np.empty((len(degrees), len(gammas),len(lambdas)))
    for index_degree, degree in enumerate(degrees):
        for index_gamma, gamma in enumerate(gammas):
            for index_lambda, lambda_ in enumerate(lambdas):
                loss_te_tmp = 0
                for k in range(k_fold):
                    _, loss_te, _ = cross_validation_log_len(y, tX, k_indices, k, degree, lambda_, gamma,log)
                    loss_te_tmp = loss_te_tmp + loss_te
                rmse_te_tmp[index_degree, index_gamma, index_lambda]= np.sqrt(2 * abs(loss_te_tmp) / k_fold)
            print("Done Lambda")
        print("Done Gamma")
    print("Done Deg")
    rmse_te = np.nanmin(rmse_te_tmp)
    print(rmse_te_tmp.shape)
    print(rmse_te_tmp[0,0,1])
    Ind_best_param = np.where(rmse_te_tmp == np.nanmin(rmse_te_tmp))
    print(Ind_best_param)
    BestDeg = degrees[np.squeeze(Ind_best_param[0])]
    BestGamma = degrees[np.squeeze(Ind_best_param[1])]
    BestLambda = degrees[np.squeeze(Ind_best_param[2])]

    return rmse_te, BestDeg, BestLambda, BestGamma



def separate_dataset(tX, ids, y = None):
    tX_list = []
    y_list = []
    ids_list = []
    for i in range(3):
        if i < 2:
            indices = np.isclose(tX[:,22], i)
        else:
            indices = np.any((np.isclose(tX[:,22], i), np.isclose(tX[:,22], i+1)), axis = 0)
        tX_list.append(tX[indices])
        ids_list.append(ids[indices])

        tX_list[i] = np.delete(tX_list[i], 22, axis=1) #Delete 22nd column
        tX_list[i] = tX_list[i][:, ~np.all(tX_list[i][1:] == tX_list[i][:-1], axis=0)] #Delete column with all the same values (so the columns of -999)

        mean = np.mean(tX_list[i], axis = 0, where = tX_list[i] != -999)
        tX_with_NaN=np.where(tX_list[i] == -999, np.nan, tX_list[i])
        median = np.nanmedian(tX_with_NaN, axis = 0)

        tX_list[i] = np.where(tX_list[i] == -999, median, tX_list[i])

        #tX_list[i] = normalize(tX_list[i])
        tX_list[i] = standardize(tX_list[i])


        if y is not None:
            y_list.append(y[indices])
    if y is not None:
        return tX_list, ids_list, y_list
    return tX_list, ids_list

def separated_train(tX_list, y_list, function, args):
    weights = []
    loss = []
    for i in range(3):
        w, l = function(y_list[i], tX_list[i], args)
        weights.append(w)
        loss.append(l)
    return weights, loss

def separated_eval(weights_list, tX_test_list):
    y_pred_list = []
    for i in range(3):
        y_pred_list.append(predict_labels(weights_list[i], tX_test_list[i]))
    return y_pred_list
