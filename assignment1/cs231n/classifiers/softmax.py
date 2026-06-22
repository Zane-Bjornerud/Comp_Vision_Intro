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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) #one training image ie. a vector of pixels. W is the weight matrix. dot product gives score for each class 

        # compute the probabilities in numerically stable way
        scores -= np.max(scores) #subtracts the largest score so numbers dont explode in exp. Doesn't change final probabilities 
        
        #softmax - turning scores into probabilities 
        exp_scores = np.exp(scores)
        probs = exp_scores/np.sum(exp_scores)

        #loss contribution from example i
        loss += -np.log(probs[y[i]]) #y[i] is the correct class for this image, so probability the model assigned to correct class
        #model is penalized more when it assigns low probability to correct class

        # gradient contributuon from example i
        #computes how W should change to reduce the loss. 
        for j in range(num_classes):
            dW[:,j] += probs[j] * X[i]

        dW[:, y[i]] -= X[i]

    #average over batch
    loss /= num_train #average loss over all examples
    dW /= num_train #average gradient

    #regularization
    loss += reg * np.sum(W*W) #penalize large weights
    dW += 2*reg*W # gradient of that penalty


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


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
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape(0)

    scores = X.dot(W) #shape(N,C)

    scores -= np.max(scores, axis=1, keepdims=True) #subtract max from each row so exponentials do not blow up

    #softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    #loss
    correct_class_probs = probs[np.arrange(num_train), y]
    loss= -np.sum(np.log(correct_class_probs))
    loss /= num_train
    loss += reg*np.sum(W*W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
