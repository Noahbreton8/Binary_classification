import A2helpers as helpers
import numpy as np
import scipy.optimize as optimize
import cvxopt.solvers as solvers
from cvxopt import matrix
import pandas as pd

solvers.options['show_progress'] = False

#a)
def binomial_deviance(w, X, y, lamb):
    w0 = w[-1]
    w = w[:-1]
    term1 = (np.dot(X, w) + w0)[:, None]
    term2 = np.logaddexp(0, -y * term1)
    return np.sum(term2) + (lamb/2) * np.dot(w, w)

def minBinDev(X, y, lamb):
    n, d = X.shape
    initial_w = np.zeros(d + 1)
    res = optimize.minimize(binomial_deviance, initial_w, args=(X, y, lamb))
    return res.x[:-1], res.x[-1]

# b)
def minHinge(X, y, lamb, stabilizer=1e-5):
    n, d = X.shape
    q = np.concatenate((np.zeros(d+1), np.ones(n)))
    q = matrix(q)

    G11 = np.zeros((n,d))
    G12 = np.zeros((n,1))
    G13 = -np.eye(n)
    G1 = np.hstack((G11, G12, G13))

    G21 = -((y * np.eye(n)) @ X)
    G22 = -y
    G23 = -np.eye(n)
    G2 = np.hstack((G21, G22, G23))

    G = np.vstack((G1, G2))
    G = matrix(G)

    H = np.concatenate((-np.zeros(n), -np.ones(n)))
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
    res = res['x']
    w = np.array(res[:d]).squeeze()
    w0 = np.array(res[d])
    return w, w0

#c)
def classify(Xtest, w, w0):
    return np.sign(np.dot(Xtest, w) + w0)[:, None]

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
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = accuracy(ytrain, classify(Xtrain, w, w0))
                test_acc_hinge[i, j, r] = accuracy(ytest, classify(Xtest, w, w0))

    # Compute the mean accuracies across runs
    train_acc_bindev_mean = np.mean(train_acc_bindev, axis=2)
    test_acc_bindev_mean = np.mean(test_acc_bindev, axis=2)
    train_acc_hinge_mean = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge_mean = np.mean(test_acc_hinge, axis=2)

    # Combine binary deviance and hinge accuracies (resulting in 4x6 matrices)
    train_acc = np.concatenate((train_acc_bindev_mean, train_acc_hinge_mean), axis = 1)
    test_acc = np.concatenate((test_acc_bindev_mean, test_acc_hinge_mean), axis = 1)

    return train_acc, test_acc

# train, test = synExperimentsRegularize()
# print(train)
# print(test)

#2a
def adj_binomial_deviance(a, X, y, lamb, K):
    a0 = a[-1]
    a = a[:-1]
    term1 = (np.dot(K, a) + a0)[:, None]
    term2 = np.logaddexp(0, -y * term1)
    return np.sum(term2) + (lamb/2) * np.dot(a, np.dot(K, a))

def adjBinDev(X, y, lamb, kernel_func):
    n, d = X.shape
    K = kernel_func(X, X)
    initial_a = np.zeros(n + 1)
    res = optimize.minimize(adj_binomial_deviance, initial_a, args=(X, y, lamb, K))
    return res.x[:-1], res.x[-1]

#b
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n, d = X.shape
    q = np.concatenate((np.zeros(n+1), np.ones(n)))
    q = matrix(q)

    K = kernel_func(X, X)

    G11 = np.zeros((n,n))
    G12 = np.zeros((n,1))
    G13 = -np.eye(n)
    G1 = np.hstack((G11, G12, G13))

    G21 = -((y * np.eye(n)) @ K)
    G22 = -y
    G23 = -np.eye(n)
    G2 = np.hstack((G21, G22, G23))

    G = np.vstack((G1, G2))
    G = matrix(G)

    H = np.concatenate((np.zeros(n), -np.ones(n)))
    H = matrix(H)

    P11 = K * lamb #* np.eye(n)
    P12 = np.zeros((n, 1))  
    P13 = np.zeros((n, n))      
    P21 = np.zeros((1, n))  
    P22 = np.zeros((1, 1))  
    P23 = np.zeros((1, n))  
    P31 = np.zeros((n, n))  
    P32 = np.zeros((n, 1))  
    P33 = np.zeros((n, n))  

    P1 = np.concatenate((P11,P12,P13), axis=1)
    P2 = np.concatenate((P21,P22,P23), axis=1)
    P3 = np.concatenate((P31,P32,P33), axis=1)
    P = np.concatenate((P1,P2,P3), axis = 0)

    P = P + stabilizer * np.eye(n+n+1)
    P = matrix(P)

    res = solvers.qp(P, q, G, H)
    res = res['x']
    a = np.array(res[:n]).squeeze()
    a0 = np.array(res[n])
    return a, a0 

def adjClassify(Xtest, a, a0, X, kernel_func):
    return np.sign(kernel_func(Xtest, X) @ a + a0)[:, None]

def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [helpers.linearKernel,
                    lambda X1, X2: helpers.polyKernel(X1, X2, 2),
                    lambda X1, X2: helpers.polyKernel(X1, X2, 3),
                    lambda X1, X2: helpers.gaussKernel(X1, X2, 1.0),
                    lambda X1, X2: helpers.gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = helpers.generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = helpers.generateData(n=n_test, gen_model=gen_model)
                
                a, a0 = adjBinDev(Xtrain, ytrain, lamb, kernel)
                train_acc_bindev[i, j, r] = accuracy(ytrain, adjClassify(Xtrain, a, a0, Xtrain, kernel))
                test_acc_bindev[i, j, r] = accuracy(ytest, adjClassify(Xtest, a, a0, Xtrain, kernel))
                
                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = accuracy(ytrain, adjClassify(Xtrain, a, a0, Xtrain, kernel))
                test_acc_hinge[i, j, r] = accuracy(ytest, adjClassify(Xtest, a, a0, Xtrain, kernel))

    # Compute the mean accuracies across runs
    train_acc_adj_bindev_mean = np.mean(train_acc_bindev, axis=2)
    test_acc_adj_bindev_mean = np.mean(test_acc_bindev, axis=2)
    train_acc_adj_hinge_mean = np.mean(train_acc_hinge, axis=2)
    test_acc_adj_hinge_mean = np.mean(test_acc_hinge, axis=2)
    
    # TODO: compute the average accuracies over runs
    # TODO: combine accuracies (bindev and hinge)
    # TODO: return 5-by-6 train accuracy and 5-by-6 test accuracy

    # Combine binary deviance and hinge accuracies (resulting in 4x6 matrices)
    train_acc = np.concatenate((train_acc_adj_bindev_mean, train_acc_adj_hinge_mean), axis = 1)
    test_acc = np.concatenate((test_acc_adj_bindev_mean, test_acc_adj_hinge_mean), axis = 1)

    return train_acc, test_acc 

# train, test = synExperimentsKernel()
# print(train)
# print(test)

#3a
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    #print(X)
    n, d = X.shape
    K = kernel_func(X, X)
    deltay = y * np.eye(n)

    q = -np.ones((n,1))
    #print(q.shape)
    q = matrix(q)

    P = (1/lamb) * deltay @ K @ deltay
    P = P + stabilizer * np.eye(n)
    #print(P.shape)
    P = matrix(P)

    h = np.concatenate((np.zeros((n,1)), np.ones((n,1))))
    #print(h.shape)
    h = matrix(h)

    G1 = -np.eye(n)
    G2 = np.eye(n)
    G = np.vstack((G1, G2))
    #print(G.shape)
    G = matrix(G)

    b = np.array([[0.]])  
    b = matrix(b)

    A = np.array(y.T, dtype=float)
    #print(A)
    A = matrix(A)

    res = solvers.qp(P, q, G, h, A, b) # A, b
    res = res['x']
    a = np.array(res[:n])

    index = np.where(a)[0]
    if index.size == 0:  # Check if the index is empty
        raise ValueError("Bad index")
        
    #print(index)

    index = index[np.argmin(np.abs(a[index]-0.5))]

    b = y[index] - ((1/lamb) * np.dot(np.dot(K[index, :], deltay), a))
    # print(b)
    # print(y[index])

    return a, b

def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    # print(a.shape)
    # print(b.shape)
    # print(X.shape)
    # print(y.shape)
    # print(Xtest.shape)
    
    K = kernel_func(Xtest, X)
    deltay = y*np.eye(len(y))
    # print(deltay.shape)
    # print(K.shape)
    yhat = (1 / lamb) * (K @ deltay @ a) + b
    return np.sign(yhat)


def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1
    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])
    # TODO: perform any necessary setup
    k_sectionsize = X.shape[0]//k

    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                Xtrain = np.concatenate((X[:l*k_sectionsize],X[(l+1)*k_sectionsize:]))# TODO: obtain the training input
                ytrain = np.concatenate((y[:l*k_sectionsize], y[(l+1)*k_sectionsize:]))# TODO: obtain the corresponding training label
                Xval = X[l*k_sectionsize:(l+1)*k_sectionsize] # TODO: obtain the validation input
                yval = y[l*k_sectionsize:(l+1)*k_sectionsize]# TODO: obtain the corresponding validation label
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                cv_acc[l, i, j] = accuracy(yval, yhat)# TODO: calculate validation accuracy
    

    # TODO: compute the average accuracies over k folds
    mean_cv_acc = np.mean(cv_acc, axis=0)

    # TODO: identify the best lamb and kernel function
    index = np.where(mean_cv_acc == np.max(mean_cv_acc))
    #print(index)
    best_lamb = lamb_list[index[0][0]]
    best_kernel = kernel_list[index[1][0]]
    # TODO: return a "len(lamb_list)-by-len(kernel_list)" accuracy variable, the best lamb and the best kernel
    return mean_cv_acc, best_lamb, index[1][0]

# kernel_list = [helpers.linearKernel,
#                     lambda X1, X2: helpers.polyKernel(X1, X2, 2),
#                     # lambda X1, X2: helpers.polyKernel(X1, X2, 3),
#                     # lambda X1, X2: helpers.gaussKernel(X1, X2, 1.0),
#                     lambda X1, X2: helpers.gaussKernel(X1, X2, 0.5)]

# lamb_list = [0.01, 0.1] 

# cv_acc, best_lamb, best_kernel = cvMnist("data", lamb_list, kernel_list)
# print(cv_acc)
# print(best_lamb)
# print(best_kernel)