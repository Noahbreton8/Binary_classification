import A2helpers as helpers
import numpy as np
import scipy.optimize as optimize
import cvxopt.solvers as solvers
import math
from cvxopt import matrix

solvers.options['show_progress'] = False

#a)
# def minBinDev(X, y, lamb):
#     n,d = X.shape
#     def binomial_deviance(w):
        
#         w0 = w[d]
#         w = w[:d]
#         term1 = -y * (X @ w + w0)
#         term2 = np.logaddexp(0, term1)
#         term3 = np.sum(term2) + lamb/2 * (w.T @ w)
#         return term3
    
#     initial_w = np.zeros(d+1)
#     res = optimize.minimize(binomial_deviance, initial_w)
#     w = res.x[:d]
#     w0 = res.x[d]
#     #print(w, w0)
#     return w, w0

def binomial_deviance(w, X, y, lamb, d):
    w0 = w[d]
    w = w[:d]
    term1 = -y * (X @ w + w0)  
    term2 = np.logaddexp(0, term1) 
    loss = np.sum(term2) + (lamb / 2) * (w.T @ w)
    return loss

def minBinDev(X, y, lamb):
    n, d = X.shape
    initial_w = np.zeros(d + 1)
    res = optimize.minimize(binomial_deviance, initial_w, args=(X, y, lamb, d))
    w = res.x[:d]
    w0 = res.x[d]
    
    return w, w0


# b)
def minHinge(X, y, lamb, stabilizer=1e-5):
    n = X.shape[0]
    d = X.shape[1]
    q = np.concatenate((np.zeros(d+1), np.ones(n)), axis=None)
    q = matrix(q)

    G11 = np.zeros_like(X)
    G12 = np.zeros((n,1))
    G13 = -np.eye(n)
    G1 = np.concatenate((G11, G12, G13), axis = 1)

    G21 = -((y * np.eye(n)) @ X)
    G22 = -y
    G23 = -np.eye(n)
    G2 = np.concatenate((G21, G22, G23), axis = 1)

    G = np.concatenate((G1,G2), axis = 0)
    G = matrix(G)

    h1 = -np.zeros(n)
    h2 = -np.ones(n)

    H = np.concatenate((h1, h2), axis = 0)
    H = matrix(H)

    P11 = lamb * np.eye(d)
    P12 = np.zeros((d, 1))  
    P13 = np.zeros((d, n))      
    P21 = np.zeros((1, d))  
    P22 = np.zeros((1, 1))  
    P23 = np.zeros((1, n))  
    P31 = np.zeros((n, d))  
    P32 = np.zeros((n, 1))  
    P33 = np.zeros((n, n))  

    P1 = np.concatenate((P11,P12,P13), axis=1)
    P2 = np.concatenate((P21,P22,P23), axis=1)
    P3 = np.concatenate((P31,P32,P33), axis=1)
    P = np.concatenate((P1,P2,P3), axis = 0)

    P = P + stabilizer * np.eye(n+d+1)
    P = matrix(P)

    res = solvers.qp(P, q, G, H)
    # print(res)
    w = res["x"][:d]
    w0 = res["x"][d]
    # print(w, w0)
    # print(res["x"], len(res["x"]),d)
    return w, w0

# c)
def classify(Xtest, w, w0):
    #print(Xtest.shape, w.shape)
    return np.sign(Xtest @ w + w0)

def accuracy(y, yhat):
    return np.mean(y == yhat)


#d)
def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    
    # Initialize 3D arrays to store accuracies over runs
    train_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = helpers.generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = helpers.generateData(n=n_test, gen_model=gen_model)
                
                # Compute accuracies for binary deviance loss
                w, w0 = minBinDev(Xtrain, ytrain, lamb)
                train_acc_bindev[i, j, r] = accuracy(ytrain, classify(Xtrain, w, w0))
                test_acc_bindev[i, j, r] = accuracy(ytest, classify(Xtest, w, w0))
                
                # Compute accuracies for hinge loss
                # w, w0 = minHinge(Xtrain, ytrain, lamb)
                # train_acc_hinge[i, j, r] = accuracy(ytrain, classify(Xtrain, w, w0))
                # test_acc_hinge[i, j, r] = accuracy(ytest, classify(Xtest, w, w0))

    # Compute the mean accuracies across runs
    train_acc_bindev_mean = np.mean(train_acc_bindev, axis=2)
    test_acc_bindev_mean = np.mean(test_acc_bindev, axis=2)
    # train_acc_hinge_mean = np.mean(train_acc_hinge, axis=2)
    # test_acc_hinge_mean = np.mean(test_acc_hinge, axis=2)
    return train_acc_bindev_mean, test_acc_bindev_mean
    # Combine binary deviance and hinge accuracies (resulting in 4x6 matrices)
    
    # train_acc = np.concatenate((train_acc_bindev_mean, train_acc_hinge_mean), axis = 1)
    # test_acc = np.concatenate((test_acc_bindev_mean, test_acc_hinge_mean), axis = 1)

    # return train_acc, test_acc

train, test = synExperimentsRegularize()
print(train)
print(test)

#e)
#Accuracy is decreasing as lamba is increasing, the model is getting more complex, and is overfitting the training data, resulting in worse test data results
#
#
#
#
#