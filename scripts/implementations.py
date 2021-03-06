import numpy as np
from proj1_helpers import *
from helpers_data import *

##################### LOSSES #####################
def compute_loss(y, tx, w):
    """Calculate the MSE loss."""
    loss = ((y - tx.dot(w))**2).sum()/(2*len(y))   #MSE
    return loss

def logsig(x):
    '''Compute the log-sigm function component-wise based on Mächler, Martin'''
    logsig = np.zeros_like(x)
    index0 = x < -33
    logsig[index0] = x[index0]
    index1 = (x >= -33) & (x < -18)
    logsig[index1] = x[index1] - np.exp(x[index1])
    index2 = (x >= -18) & (x < 37)
    logsig[index2] = -np.log1p(np.exp(-x[index2]))
    index3 = x >= 37
    logsig[index3] = -np.exp(-x[index3])
    return logsig

def compute_loss_log(y, tx, w):
    '''Compute the loss function for a logistic model'''
    z = np.dot(tx, w)
    y = np.asarray(y)
    return np.mean((1 - y) * z - logsig(z))


##################### GRADIENTS #####################
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    grad = -tx.T.dot(y - tx.dot(w))/len(y)
    return grad

def compute_sigy(x, y):
    ''' Compute sig(x)-y composent-wise with sig(x) the sigmoid function based on Mächler, Martin'''
    index = x < 0
    result = np.zeros_like(x)
    exp_x = np.exp(x[index])
    y_index = y[index]
    result[index] = ((1 - y_index) * exp_x - y_index) / (1 + exp_x)
    exp_nx = np.exp(-x[~index])
    y_nidx = y[~index]
    result[~index] = ((1 - y_nidx) - y_nidx * exp_nx) / (1 + exp_nx)
    return result

def compute_gradient_log(y, tx, w):
    z = tx.dot(w)
    s = compute_sigy(z, y)
    return tx.T.dot(s) / tx.shape[0]


##################### BATCH #####################

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


##################### METHODS #####################
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    loss_prev = 0
    threshold = 1e-8

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - (gamma * gradient)
        # if n_iter % 100 == 0:
            # print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        if n_iter > 1 and np.abs(loss - loss_prev) < threshold:
            break
        if loss == np.inf or loss == np.nan:
            # print('loss error')
            break
        loss_prev = loss
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - (gamma * gradient)
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_loss(y, tx, w)
    return w, mse 


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_pr = lambda_ * 2 * len(y)
    w = np.linalg.solve(tx.T @ tx + lambda_pr * np.eye(tx.shape[1]), tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss



##################### LOGISTIC REGRESSION  #####################

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression."""
    threshold = 1e-7
    w = initial_w
    loss_prev = 0
    y = y.reshape((-1,1))

    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_loss_log(y, tx, w)
        grad = compute_gradient_log(y, tx, w)
        w -= gamma * grad

        # log info
        # if iter % 100 == 0:
        #   print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        # converge criterion
        if iter > 1 and np.abs(loss - loss_prev) < threshold:
            break  
        loss_prev = loss

    return w, loss 


def reg_logistic_regression(y, tx, initial_w, max_iter, lambda_, gamma):
    """implement regularized logistic regression."""
    # init parameters
    threshold = 1e-7
    loss_prev = 0

    w = initial_w
    y = y.reshape((-1,1))
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss = compute_loss_log(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = compute_gradient_log(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        
        # log info
        # if iter % 100 == 0:
        #     print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        if loss < 0:
            print('NEGATIVE LOSS')
            break
        # converge criterion
        if iter > 1 and np.abs(loss - loss_prev) < threshold:
            break
        loss_prev = loss
    return w, loss


##################### GRID SEARCH #####################


def grid_search(y, tX, function, k_fold = 4, degrees = range(1, 15), lambdas = np.arange(1), gammas = np.arange(1), dataset = 0):
    """Find the best hyper parameter for a given model using k-fold cross validation."""
    k_indices = build_k_indices(y, k_fold)
    score_grid = np.empty((len(degrees), len(gammas),len(lambdas)))

    for index_degree, degree in enumerate(degrees):
        for index_gamma, gamma in enumerate(gammas):
            for index_lambda, lambda_ in enumerate(lambdas):
                score_tmp = 0
                for k in range(k_fold):
                    _, score_te, _ = cross_validation(y, tX, k_indices, k, degree, function, (lambda_, gamma), dataset)
                    score_tmp = score_tmp + score_te
                score_grid[index_degree, index_gamma, index_lambda]= score_tmp / k_fold
    best_score = np.nanmax(score_grid)
    Ind_best_param = np.where(score_grid == best_score)
    BestDeg = degrees[np.squeeze(Ind_best_param[0][0])]
    BestGamma = gammas[np.squeeze(Ind_best_param[1][0])]
    BestLambda = lambdas[np.squeeze(Ind_best_param[2][0])]
    return best_score, BestDeg, BestLambda, BestGamma


##################### CROSS VALIDATION #####################

def cross_validation(y, x, k_indices, k, degree, function, args = None, dataset = 0):
    """Return the score of a model on training sample to cross-validate hyper parameters."""

    indices_te = k_indices[k]
    indices_tr = np.delete(k_indices, k, axis=0)
    indices_tr = np.concatenate(indices_tr, axis= None)
    x_tr = x[indices_tr]
    y_tr = y[indices_tr]
    x_te = x[indices_te]
    y_te = y[indices_te]
    
    x_tr_poly, x_te_poly = build_poly_log(x_tr, degree, x_te, dataset)

    if (function == 1):
        max_iter= 300
        initial_w = np.zeros(x_tr_poly.shape[1])
        weights, loss_tr = least_squares_GD(y_tr, x_tr_poly, initial_w, max_iter, args[1])
        score = compute_score(y_te, x_te_poly, weights)

    elif (function == 2):
        max_iter= 1000
        initial_w = np.zeros(x_tr_poly.shape[1])
        weights, loss_tr = least_squares_SGD(y_tr, x_tr_poly, initial_w, max_iter, args[1])
        score = compute_score(y_te, x_te_poly, weights)

    elif (function == 3):
        weights, loss_tr = least_squares(y_tr, x_tr_poly)
        #loss_te = compute_loss(y_te, x_te_poly, weights)
        score = compute_score(y_te, x_te_poly, weights)

    elif (function == 4):
        weights, loss_tr = ridge_regression(y_tr, x_tr_poly, args[0])
        #loss_te = compute_loss(y_te, x_te_poly, weights)
        score = compute_score(y_te, x_te_poly, weights)
    
    elif (function == 5):
        max_iter= 3000
        initial_w = np.zeros((x_tr_poly.shape[1], 1))
        weights, loss_tr = logistic_regression(y_tr, x_tr_poly, initial_w, max_iter, args[1])
        #loss_te = compute_loss_log(y_te, x_te_poly, weights)
        score = compute_score(y_te, x_te_poly, weights)

    elif (function == 6):
        max_iter= 3000
        initial_w = np.zeros((x_tr_poly.shape[1], 1))
        weights, loss_tr = reg_logistic_regression(y_tr, x_tr_poly, initial_w, max_iter, args[0], args[1])
        #loss_te = compute_loss_log(y_te, x_te_poly, weights)
        score = compute_score(y_te, x_te_poly, weights)

    else:
        print('error function name')

    return loss_tr, score, weights


##################### EVAL #####################


def separated_eval(weights_list, tX_test_list):
    y_pred_list = []
    for i in range(6):
        y_pred_list.append(predict_labels(weights_list[i], tX_test_list[i]))

    return y_pred_list

