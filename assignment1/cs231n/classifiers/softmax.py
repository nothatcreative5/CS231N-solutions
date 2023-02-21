from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores =  X @ W 
    for i in range(X.shape[0]):
      row = scores[i]
      row = row - np.max(row)
      loss += np.log(np.sum(np.exp(row))) - row[y[i]]
    loss /= X.shape[0]
    loss += reg * 1/2 * np.sum(np.square(W))



    # couldn't be bothered
    scores = scores - np.max(scores, axis = 1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs,axis = 1, keepdims=True)
    probs[np.arange(X.shape[0]),y] -= 1
    dW = X.T @ probs
    dW /= X.shape[0]
    dW += reg * W * 2

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores =  X @ W
    scores = scores - np.max(scores, axis = 1, keepdims = True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs,axis = 1, keepdims=True)
    loss = -np.sum(np.log(probs[np.arange(X.shape[0]), y].reshape(-1,1)))
    loss /= X.shape[0]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    probs[np.arange(X.shape[0]),y] -= 1
    dW = X.T @ probs
    dW /= X.shape[0]
    dW += reg * W * 2

    return loss, dW
