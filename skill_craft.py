import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def read_csv_enhanced(filename, columns):
    """
    Read CSV file.

    :param filename: name of the file to read
    :param columns: name of columns to read
    :return: two tensors, the second one contains the first column data, the first one all the other columns
    """
    print("\nREADING CSV %s ..." % filename)
    data = pd.read_csv(filename)
    y = torch.tensor(data[columns[0]])
    x = torch.empty(len(data[columns[0]]), len(columns) - 1)
    for n, column in enumerate(columns[1:]):
        print("column %s '%s'" % (n, column))
        x[:,n] = torch.tensor(data[column])
    return x, y


def create_test_sets(x_tensor, apm_tensor):
    print("\nCREATING TEST SETS ...")
    # build Set 1 with the first 30 entries (*** TEST SET 1 ***)
    x_tensor_1 = x_tensor[:30]
    y_tensor_1 = apm_tensor[:30]
    print("x_tensor_1 (%s): %s" % (x_tensor_1.shape, x_tensor_1))
    print("y_tensor_1 (%s): %s" % (y_tensor_1.shape, y_tensor_1))

    # build an intermediate set with all remaining records beyond index 30
    remaining_x_tensor = x_tensor[30:]
    remaining_y_tensor = apm_tensor[30:]

    # build Set 2 with 20% of remaining data (*** TEST SET 2 ***)
    # calculate the size of the remaining data  and the index where its first 20% ends
    remaining_size = remaining_x_tensor.shape[0]
    twenty_percent_index = (remaining_size * 20) // 100

    # create random permutation if integers from 0 to remaining_size
    random_perm = torch.randperm(remaining_size)
    print("\nrandom_perm (%s): %s" % (random_perm.shape, random_perm))
    #print("remaining_x_tensor (%s): %s" % (remaining_x_tensor.shape[0], remaining_x_tensor))

    x_tensor_2 = remaining_x_tensor[random_perm[:twenty_percent_index]]
    y_tensor_2 = remaining_y_tensor[random_perm[:twenty_percent_index]]
    print("\nx_tensor_2 (%s): %s ..." % (x_tensor_2.shape, x_tensor_2[:10]))
    print("y_tensor_2 (%s): %s ..." % (y_tensor_2.shape, y_tensor_2[:10]))

    # build Set 3 with 80% of remaining data (*** TEST SET 3 / TRAINING SET ***)
    x_tensor_3 = remaining_x_tensor[random_perm[twenty_percent_index:]]
    y_tensor_3 = remaining_y_tensor[random_perm[twenty_percent_index:]]
    print("\nx_tensor_3 (%s): %s" % (x_tensor_3.shape, x_tensor_3))
    print("y_tensor_3 (%s): %s" % (y_tensor_3.shape, y_tensor_3))

    return x_tensor_1, y_tensor_1, x_tensor_2, y_tensor_2, x_tensor_3, y_tensor_3


def ols(tensor_x, tensor_y):
    """
    Calculate Ordinary Least Squares for a function of the form y = wx + b.

    :param tensor_x: tensor X
    :param tensor_y: tensor Y
    :return: a tuple with two values: w and b
    """
    # calculate values using matrices like w=(XT*X)-1*XT*y
    # build new X with additional 1s
    print("\nOLS")
    print("x (%s): %s" % (tensor_x.shape, tensor_x))
    print("y (%s): %s" % (tensor_y.shape, tensor_y))

    x_ones = torch.ones((tensor_x.shape[0], 1))
    x_reshaped = tensor_x.reshape((tensor_x.shape[0], 1))
    x_prime = torch.cat((x_reshaped, x_ones), 1)
    print("x_prime: %s" % x_prime)

    matmul_res1 = torch.matmul(x_prime.T, x_prime)
    print("matmul_res1: %s" % matmul_res1)
    matmul_res1_inverse = matmul_res1.inverse()
    print("matmul_res1_inverse: %s" % matmul_res1_inverse)
    matmul_res2 = torch.matmul(matmul_res1_inverse, x_prime.T)
    print("matmul_res2 (%s, %s): %s" % (matmul_res2.shape[0], matmul_res2.shape[1], matmul_res2))
    matmul_res_final = torch.matmul(matmul_res2, tensor_y)
    print("matmul_res_final: %s" % matmul_res_final)
    
    return matmul_res_final


def plot_data_set_and_function(al_tensor, apm_tensor, fn_x_tensor=None, fn_y_tensor=None, name="Figure"):
    """
    Plot a given test set.

    :param al_tensor: tensor X
    :param apm_tensor: tensor y
    :param fn_x_tensor: tensor x for function line plotting
    :param fn_y_tensor: tensor y for function line plotting
    :param name: figure name
    """
    print("\n*** plot_data_set_and_model: %s ***" % name)
    plt.figure(num=name)
    plt.scatter(al_tensor, apm_tensor)
    if fn_y_tensor is not None:
        x_tensor = al_tensor
        if fn_x_tensor is not None:
            x_tensor = fn_x_tensor
        plt.plot(x_tensor, fn_y_tensor, color="red")
    plt.xlabel("Action Latency")
    plt.ylabel("APM")
    plt.show()


def model(tensor_x, tensor_w, tensor_b):
    """
    Model a linear function of the form y = wx + b.

    :param tensor_x: x tensor
    :param tensor_w: w parameter tensor
    :param tensor_b: v parameter tensor
    :return: the value for y where y = wx + b
    """
    tensor_y = tensor_w * tensor_x
    #print("tensor_y (%s): %s" % (tensor_y.shape, tensor_y))
    tensor_y = tensor_y + tensor_b
    #print("tensor_y (%s): %s" % (tensor_y.shape, tensor_y))
    return tensor_y


def model_nonlin(tensor_x, tensor_a, tensor_b, tensor_c):
    """
    Model a non-linear function of the form y = a * x**b + c.

    :param tensor_x: x tensor
    :param tensor_a: a parameter tensor
    :param tensor_b: b parameter tensor
    :param tensor_c: c parameter tensor
    :return: the value for the function given the inputs.
    """
    tensor_y = tensor_a * (tensor_x ** tensor_b)
    #print("tensor_y (%s): %s" % (tensor_y.shape, tensor_y))
    tensor_y = tensor_y + tensor_c
    #print("tensor_y (%s): %s" % (tensor_y.shape, tensor_y))
    return tensor_y


def model_nonlin2(tensor_x, tensor_a, tensor_b, tensor_c):
    """
    Model a non-linear function of the form y = a * e**(b*x) + c.

    :param tensor_x: x tensor
    :param tensor_a: a parameter tensor
    :param tensor_b: b parameter tensor
    :param tensor_c: c parameter tensor
    :return: the value for the function given the inputs.
    """
    tensor_y = tensor_a * torch.exp(tensor_b * tensor_x)
    #print("tensor_y (%s): %s" % (tensor_y.shape, tensor_y))
    tensor_y = tensor_y + tensor_c
    #print("tensor_y (%s): %s" % (tensor_y.shape, tensor_y))
    return tensor_y


def loss_fn(tensor_y, tensor_real_y):
    """
    Takes a tensor containing output values y from our model, and another tensor containing observed output values,
    and computes the mean squared distance between the two.

    :param tensor_y: output values from our model
    :param tensor_real_y: observed out values
    :return: the mean squared distance between the two tensors
    """
    dif = tensor_real_y - tensor_y
    #print("dif (%s): %s" % (dif.shape, dif))
    squared_dif = dif ** 2
    #print("squared_dif (%s): %s" % (squared_dif.shape, squared_dif))
    squared_dif_mean = squared_dif.mean()
    #print("squared_dif_mean (%s): %s" % (squared_dif_mean.shape, squared_dif_mean))
    return squared_dif_mean


def dmodel_w(tensor_x):
    """
    Calculates the derivative of parameter w in a function of the form y = wx + b.

    :param tensor_x: tensor x
    :return: the value for the derivative
    """
    # or alternatively def dmodel_w(tensor_x, tensor_w, tensor_b):
    return tensor_x


def dmodel_b():
    """
    Calculates the derivative of parameter b in a function of the form y = wx + b.

    :return: the value for the derivative.
    """
    return 1


def dloss_m(tensor_y, tensor_real_y):
    """
    Calculates the derivative of the loss function for a function of the form y = wx + b.

    :return: the value for the derivative
    """
    return -2 * (tensor_real_y - tensor_y)


def normalize_tensor(tensor):
    """
    Normalize a tensor by subtracting its mean and dividing by half of its range.

    :param tensor: tensor to normalize
    :return: the normalized tensor
    """
    print("\nNormalization - Original tensor: %s" % tensor)
    print("\tmean: %s - max: %s - min %s" % (torch.mean(tensor), torch.max(tensor), torch.min(tensor)))
    norm_tensor = tensor - torch.mean(tensor)
    print("\tNum: %s" % norm_tensor)
    print("\tDen: %s" % ((torch.max(tensor) - torch.min(tensor)) / 2))
    norm_tensor = norm_tensor / ((torch.max(tensor) - torch.min(tensor)) / 2)
    print("\tnorm_tensor: %s" % norm_tensor)
    return norm_tensor


def training(iterations, tensor_w, tensor_b, alpha, tensor_x, tensor_y):
    """
    Trains a linear model by hand.

    :param iterations: number of epochs
    :param tensor_w: w parameter
    :param tensor_b: b parameter
    :param alpha: learning rate
    :param tensor_x: the independent variable
    :param tensor_y: the dependent variable
    :return: fitted values for w and b.
    """
    print("\n*** TRAINING ***")
    for it in range(iterations):
        # calculate loss
        tensor_y_calc = model(tensor_x, tensor_w, tensor_b)
        loss = loss_fn(tensor_y_calc, tensor_y)

        # calculate gradients
        gradient_w = (dloss_m(tensor_y_calc, tensor_y) * dmodel_w(tensor_x)).mean()
        gradient_b = (dloss_m(tensor_y_calc, tensor_y) * dmodel_b()).mean()

        # adjust coefficients
        tensor_w = tensor_w - alpha * gradient_w
        tensor_b = tensor_b - alpha * gradient_b

        print("N: %s\t | Loss: %f\t | Grad W: %s\t | Grad B: %s\t | W: %s\t | B: %s" %
              (it, loss, gradient_w, gradient_b, tensor_w, tensor_b))

    return tensor_w, tensor_b


def training_auto(iterations, tensor_w, tensor_b, alpha, tensor_x, tensor_y):
    """
    Trains a linear model by automatically.

    :param iterations: number of epochs
    :param tensor_w: w parameter
    :param tensor_b: b parameter
    :param alpha: learning rate
    :param tensor_x: the independent variable
    :param tensor_y: the dependent variable
    :return: fitted values for w and b.
    """
    print("\n*** TRAINING AUTO ***")
    for it in range(iterations):
        if tensor_w.grad is not None:
            tensor_w.grad.zero_()
        if tensor_b.grad is not None:
            tensor_b.grad.zero_()

        # calculate loss
        tensor_y_calc = model(tensor_x, tensor_w, tensor_b)
        loss = loss_fn(tensor_y_calc, tensor_y)
        loss.backward()

        # adjust coefficients
        tensor_w = (tensor_w - alpha * tensor_w.grad).detach().requires_grad_()
        tensor_b = (tensor_b - alpha * tensor_b.grad).detach().requires_grad_()

        print("N: %s\t | Loss: %f\t | W: %s\t | B: %s" %
              (it, loss, tensor_w, tensor_b))

    return tensor_w, tensor_b


def training_opt(iterations, tensor_w, tensor_b, tensor_x, tensor_y, optimizer):
    """
    Train a linear model automatically and using an optimizer.

    :param iterations: number of epochs
    :param tensor_w: w parameter
    :param tensor_b: b parameter
    :param tensor_x: the independent variable
    :param tensor_y: the dependent variable
    :param optimizer: the the optimizer
    :return: fitted values for w and b.
    """
    print("\n*** TRAINING OPT ***")
    for it in range(iterations):
        optimizer.zero_grad()

        # calculate loss
        tensor_y_calc = model(tensor_x, tensor_w, tensor_b)
        loss = loss_fn(tensor_y_calc, tensor_y)
        loss.backward()
        optimizer.step()

        print("N: %s\t | Loss: %f\t | W: %s\t | B: %s" %
              (it, loss, tensor_w, tensor_b))

    return tensor_w, tensor_b


def training_nonlin(iterations, tensor_a, tensor_b, tensor_c, tensor_x, tensor_y, nonlin_function, optimizer):
    """
    Train a non linear model automatically and using an optimizer.

    :param iterations: number of epochs
    :param tensor_a: a parameter
    :param tensor_b: b parameter
    :param tensor_c: c parameter
    :param tensor_x: the independent variable
    :param tensor_y: the dependent variable
    :param nonlin_function: non-linear function
    :param optimizer: the the optimizer
    :return: fitted values for a, b and c.
    """
    print("\n*** TRAINING NON LINEAR ***")
    for it in range(iterations):
        optimizer.zero_grad()

        # calculate loss
        tensor_y_calc = nonlin_function(tensor_x, tensor_a, tensor_b, tensor_c)
        loss = loss_fn(tensor_y_calc, tensor_y)
        loss.backward()
        optimizer.step()

        print("N: %s\t | Loss: %f\t | A: %s\t | B: %s\t | C: %s" %
              (it, loss, tensor_a, tensor_b, tensor_c))

    return tensor_a, tensor_b, tensor_c


def lab_1(al_tensor, apm_tensor, al1_tensor, apm1_tensor, al2_tensor, apm2_tensor, al3_tensor, apm3_tensor):
    """
    Lab 1.
    :param al_tensor: 
    :param apm_tensor: 
    :param al1_tensor: 
    :param apm1_tensor: 
    :param al2_tensor: 
    :param apm2_tensor: 
    :param al3_tensor: 
    :param apm3_tensor: 
    :return: nothing
    """
    '''
    Exploratory Data Analysis
    Take the test set 1 (with thirty entries), and plot it as a scatterplot with Matplotlib.
    Does the data look linear? Try the same with the training set.
    What are the maximum, minimum and mean values for APM and ActionLatency?
    What is the standard deviation of the two variables?
    What is the correlation between the two variables?
    '''
    # scatter plot and line in the same figure
    plot_data_set_and_function(al1_tensor, apm1_tensor, name="Test Set 1")
    plot_data_set_and_function(al3_tensor, apm3_tensor, name="Test Set 3/Training Set")

    # print some statistics
    print("")
    print("max AL: %s - APM: %s" % (torch.max(al_tensor), torch.max(apm_tensor)))
    print("min AL: %s - APM: %s" % (torch.min(al_tensor), torch.min(apm_tensor)))
    print("mean AL: %s - APM: %s" % (torch.mean(al_tensor), torch.mean(apm_tensor)))
    print("std AL: %s - APM: %s" % (torch.std(al_tensor), torch.std(apm_tensor)))
    print("corr AL-APM: %s" % (np.corrcoef(al_tensor, apm_tensor)))

    # calculate Ordinary Least Squares from the Training Set (#3)
    ols_result = ols(al3_tensor, apm3_tensor)
    print("\nOLS (%s): %s" % (ols_result.shape, ols_result))

    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_y_tensor=model(al1_tensor, ols_result[0], ols_result[1]), name="Test Set 1")
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_y_tensor=model(al2_tensor, ols_result[0], ols_result[1]), name="Test Set 2")
    plot_data_set_and_function(al3_tensor, apm3_tensor, fn_y_tensor=model(al3_tensor, ols_result[0], ols_result[1]), name="Test Set 3/Training Set")

    '''
    TRAINING
    Do n times:
        Calculate the current estimate for y using the current values for w and b, as well as the loss
        Calculate the gradient
        Print the current loss, gradient and iteration count
        Update w and b using the gradient
    '''
    tensor_w = torch.tensor([-2.0])
    tensor_b = torch.tensor([230.0])

    # TRAINING - NOT NORMALIZED
    tensor_w, tensor_b = training(1000, tensor_w, tensor_b, 1e-4, al3_tensor, apm3_tensor)
    # tensor_w, tensor_b = training(10000, tensor_w, tensor_b, 1e-4, al3_tensor, apm3_tensor)
    plot_data_set_and_function(al3_tensor, apm3_tensor, fn_y_tensor=model(al3_tensor, tensor_w, tensor_b), name="X NOT Normalized")

    # TRAINING - NORMALIZED
    al3_tensor_norm = normalize_tensor(al3_tensor)
    tensor_w_norm = tensor_w.clone().detach()
    tensor_b_norm = tensor_b.clone().detach()
    tensor_w_norm, tensor_b_norm = training(1000, tensor_w_norm, tensor_b_norm, 1e-1, al3_tensor_norm, apm3_tensor)
    plot_data_set_and_function(al3_tensor_norm, apm3_tensor, fn_y_tensor=model(al3_tensor_norm, tensor_w_norm, tensor_b_norm), name="X Normalized")

    # LOSS
    tensor_y_learned = model(al3_tensor, tensor_w, tensor_b)
    learned_model_loss = loss_fn(tensor_y_learned, apm3_tensor)
    print("\nLOSS for TEST SET 3 (Training Set): %s " % learned_model_loss)

    tensor_y_learned = model(al1_tensor, tensor_w, tensor_b)
    learned_model_loss = loss_fn(tensor_y_learned, apm1_tensor)
    print("\nLOSS for TEST SET 1: %s " % learned_model_loss)
    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_y_tensor=tensor_y_learned, name="Learned Model vs Test Set 1")

    tensor_y_learned = model(al2_tensor, tensor_w, tensor_b)
    learned_model_loss = loss_fn(tensor_y_learned, apm2_tensor)
    print("\nLOSS for TEST SET 2: %s " % learned_model_loss)
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_y_tensor=tensor_y_learned, name="Learned Model vs Test Set 2")

    '''
    Automated Gradients and Optimization
    '''
    # Auto
    tensor_w_auto = torch.tensor([-2.0], requires_grad=True)
    tensor_b_auto = torch.tensor([230.0], requires_grad=True)
    tensor_w_auto, tensor_b_auto = training_auto(1000, tensor_w_auto, tensor_b_auto, 1e-4, al3_tensor, apm3_tensor)

    model_auto = model(al1_tensor, tensor_w_auto.detach(), tensor_b_auto.detach())
    learned_model_loss = loss_fn(model_auto, apm1_tensor)
    print("\nLOSS for TEST SET 1 (Auto): %s " % learned_model_loss)
    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_y_tensor=model_auto, name="Auto - Test Set 1")

    model_auto = model(al2_tensor, tensor_w_auto.detach(), tensor_b_auto.detach())
    learned_model_loss = loss_fn(model_auto, apm2_tensor)
    print("\nLOSS for TEST SET 2 (Auto): %s " % learned_model_loss)
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_y_tensor=model_auto, name="Auto - Test Set 2")

    # Optimization: SDG
    tensor_w_opt = torch.tensor([-2.0], requires_grad=True)
    tensor_b_opt = torch.tensor([230.0], requires_grad=True)
    tensor_w_opt, tensor_b_opt = training_opt(1000, tensor_w_opt, tensor_b_opt, al3_tensor, apm3_tensor,
                                              optim.SGD([tensor_w_opt, tensor_b_opt], lr=1e-4))

    model_opt = model(al1_tensor, tensor_w_opt.detach(), tensor_b_opt.detach())
    learned_model_loss = loss_fn(model_opt, apm1_tensor)
    print("\nLOSS for TEST SET 1 (SDG): %s " % learned_model_loss)
    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_y_tensor=model_opt, name="Opt SGD - Test Set 1")

    model_opt = model(al2_tensor, tensor_w_opt.detach(), tensor_b_opt.detach())
    learned_model_loss = loss_fn(model_opt, apm2_tensor)
    print("\nLOSS for TEST SET 2 (SDG): %s " % learned_model_loss)
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_y_tensor=model_opt, name="Opt SGD - Test Set 2")

    # Optimization: Adam UNSCALED
    tensor_w_opt = torch.tensor([-2.0], requires_grad=True)
    tensor_b_opt = torch.tensor([230.0], requires_grad=True)
    tensor_w_opt, tensor_b_opt = training_opt(1000, tensor_w_opt, tensor_b_opt, al3_tensor, apm3_tensor,
                                              optim.Adam([tensor_w_opt, tensor_b_opt], lr=1e-4))

    model_opt = model(al1_tensor, tensor_w_opt.detach(), tensor_b_opt.detach())
    learned_model_loss = loss_fn(model_opt, apm1_tensor)
    print("\nLOSS for TEST SET 1 (Adam UNSCALED): %s " % learned_model_loss)
    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_y_tensor=model_opt, name="Opt Adam UNSCALED - Test Set 1")

    model_opt = model(al2_tensor, tensor_w_opt.detach(), tensor_b_opt.detach())
    learned_model_loss = loss_fn(model_opt, apm2_tensor)
    print("\nLOSS for TEST SET 2 (Adam UNSCALED): %s " % learned_model_loss)
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_y_tensor=model_opt, name="Opt Adam UNSCALED - Test Set 2")

    # Optimization: Adam SCALED
    tensor_w_opt, tensor_b_opt = training_opt(1000, tensor_w_opt, tensor_b_opt, al3_tensor_norm, apm3_tensor,
                                              optim.Adam([tensor_w_opt, tensor_b_opt], lr=1e-4))

    model_opt = model(al1_tensor, tensor_w_opt.detach(), tensor_b_opt.detach())
    learned_model_loss = loss_fn(model_opt, apm1_tensor)
    print("\nLOSS for TEST SET 1 (Adam SCALED): %s " % learned_model_loss)
    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_y_tensor=model_opt, name="Opt Adam SCALED - Test Set 1")

    model_opt = model(al2_tensor, tensor_w_opt.detach(), tensor_b_opt.detach())
    learned_model_loss = loss_fn(model_opt, apm2_tensor)
    print("\nLOSS for TEST SET 2 (Adam SCALED): %s " % learned_model_loss)
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_y_tensor=model_opt, name="Opt Adam SCALED - Test Set 2")

    '''
    A better fit
    '''
    # FIRST NON_LINEAR FUNCTION: y = a * x**b + c
    tensor_a = torch.tensor([14000.0], requires_grad=True)
    tensor_b = torch.tensor([-1.0], requires_grad=True)
    tensor_c = torch.tensor([-20.0], requires_grad=True)

    optimizer = optim.Adam([tensor_a, tensor_b, tensor_c], lr=1e-4)
    tensor_a, tensor_b, tensor_c = training_nonlin(1000, tensor_a, tensor_b, tensor_c, al3_tensor, apm3_tensor,
                                                   model_nonlin, optimizer)

    model_better = model_nonlin(al1_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    learned_model_loss = loss_fn(model_better, apm1_tensor)
    print("\nLOSS for TEST SET 1 (Better Fit F1): %s " % learned_model_loss)
    fn_x_tensor = np.linspace(al1_tensor.min(), al1_tensor.max(), 1000)
    fn_y_tensor = model_nonlin(fn_x_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_x_tensor, fn_y_tensor, name="Better fit: y = a* x**b + c - Test Set 1")

    model_better = model_nonlin(al2_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    learned_model_loss = loss_fn(model_better, apm2_tensor)
    fn_x_tensor = np.linspace(al2_tensor.min(), al2_tensor.max(), 1000)
    fn_y_tensor = model_nonlin(fn_x_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    print("\nLOSS for TEST SET 2 (Better Fit F1): %s " % learned_model_loss)
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_x_tensor, fn_y_tensor, name="Better fit: y = a* x**b + c - Test Set 2")

    # SECOND NON_LINEAR FUNCTION: y = a * e**(b*x) + c
    tensor_a = torch.tensor([1400.0], requires_grad=True)
    tensor_b = torch.tensor([-0.05], requires_grad=True)
    tensor_c = torch.tensor([20.0], requires_grad=True)

    optimizer = optim.Adam([tensor_a, tensor_b, tensor_c], lr=1e-5)
    tensor_a, tensor_b, tensor_c = training_nonlin(1000, tensor_a, tensor_b, tensor_c, al3_tensor, apm3_tensor,
                                                   model_nonlin2, optimizer)

    model_better = model_nonlin2(al1_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    learned_model_loss = loss_fn(model_better, apm1_tensor)
    print("\nLOSS for TEST SET 1 (Better Fit F2): %s " % learned_model_loss)
    fn_x_tensor = np.linspace(al1_tensor.min(), al1_tensor.max(), 1000)
    fn_y_tensor = model_nonlin2(fn_x_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    plot_data_set_and_function(al1_tensor, apm1_tensor, fn_x_tensor, fn_y_tensor, name="Better fit: y = a* e**(b*x) + c - Test Set 1 ")

    model_better = model_nonlin2(al2_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    learned_model_loss = loss_fn(model_better, apm2_tensor)
    print("\nLOSS for TEST SET 2 (Better Fit F2): %s " % learned_model_loss)
    fn_x_tensor = np.linspace(al2_tensor.min(), al2_tensor.max(), 1000)
    fn_y_tensor = model_nonlin2(fn_x_tensor, tensor_a.detach_(), tensor_b.detach(), tensor_c.detach())
    plot_data_set_and_function(al2_tensor, apm2_tensor, fn_x_tensor, fn_y_tensor, name="Better fit: y = a* e**(b*x) + c - Test Set 2 ")


''' -------------------------- LAB 2 -------------------------------------- '''


class SkillCraftNN(nn.Module):

    def __init__(self, input_n, hidden_n, output_n, activation_fn):
        super(SkillCraftNN, self).__init__()
        print("\n*** SkillCraftNN i: %s - h: %s - o: %s ***" % (input_n, hidden_n, output_n))
        self.hidden_linear = nn.Linear(input_n, hidden_n)
        self.hidden_activation = activation_fn
        self.output_linear = nn.Linear(hidden_n, output_n)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t


def train_nn(iterations, nn_model, optimizer, nn_loss_fn, tensor_x, tensor_y, input_n, output_n):
    print("\n*** TRAINING NN ***")
    print("\ntensor_x (%s): %s" % (tensor_x.shape, tensor_x))
    tensor_x_reshaped = tensor_x.view(-1, input_n)
    print("\ntensor_x_reshaped (%s): %s" % (tensor_x_reshaped.shape, tensor_x_reshaped))

    for it in range(1, iterations + 1):
        tensor_y_pred = nn_model(tensor_x_reshaped)
        tensor_y_pred_reshaped = tensor_y_pred.view(-1, output_n)
        loss = nn_loss_fn(tensor_y.view(-1, output_n), tensor_y_pred_reshaped)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 100 == 0: print("N: %s\t | Loss: %f\t" % (it, loss))


def nn_model_test_set(al_tensor, apm_tensor, nn_model, nn_loss_fn, test_set_name):
    model_y = nn_model(al_tensor.view(-1, 1))
    model_loss = nn_loss_fn(model_y, apm_tensor.view(-1, 1))
    print("\nLOSS for %s (NN1): %s " % (test_set_name, model_loss))


def nn_plot_test_test(al_tensor, apm_tensor, nn_model, test_set_name):
    fn_x_tensor = torch.tensor(np.linspace(al_tensor.min(), al_tensor.max(), 1000), dtype=torch.float)
    fn_y_tensor = nn_model(fn_x_tensor.view(-1, 1))
    plot_data_set_and_function(al_tensor, apm_tensor, fn_x_tensor, fn_y_tensor.detach().numpy(), test_set_name)


def lab_2(al_tensor, apm_tensor, al1_tensor, apm1_tensor, al2_tensor, apm2_tensor, al3_tensor, apm3_tensor):
    """
    Lab 2.
    :param al_tensor:
    :param apm_tensor:
    :param al1_tensor:
    :param apm1_tensor:
    :param al2_tensor:
    :param apm2_tensor:
    :param al3_tensor:
    :param apm3_tensor:
    :return: nothing.
    """
    ''' 
    LINEAR NN:
    Update your code to use torch.nn.Linear(1,1) instead of your own model function (the parameters are the number of 
    inputs and outputs). Instead of w and b, the neural network gives you a parameters() method, which you should pass 
    to your optimizer. Check if the results are the same, better or worse than your results from last week. 
    '''
    nn_model = nn.Linear(1, 1)
    # LAB NOTE: 1e-4 is too slow, 1e-1 is a lot better
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-1)
    nn_loss_fn = nn.MSELoss()

    # LAB NOTE: 1000 iterations is not enough (Loss: 3839.005127), 5000 seems good  (Loss: 1302.502563)
    train_nn(5000, nn_model, optimizer, nn_loss_fn, al3_tensor, apm3_tensor, 1, 1)

    # model each Test Set and calculate loss
    nn_model_test_set(al3_tensor, apm3_tensor, nn_model, nn_loss_fn, "NN1 - TEST SET 3")
    nn_plot_test_test(al3_tensor, apm3_tensor, nn_model, "NN1 - TEST SET 3")

    nn_model_test_set(al1_tensor, apm1_tensor, nn_model, nn_loss_fn, "NN1 - TEST SET 1")
    nn_plot_test_test(al1_tensor, apm1_tensor, nn_model, "NN1 - TEST SET 1")

    nn_model_test_set(al2_tensor, apm2_tensor, nn_model, nn_loss_fn, "NN1 - TEST SET 2")
    nn_plot_test_test(al2_tensor, apm2_tensor, nn_model, "NN1 - TEST SET 2")

    ''' 
    NEW NEURAL NETWORK (using nn.Module)
    For the purpose of this lab define a simple neural network with a few neurons in one hidden layer, and try several 
    different numbers of hidden neurons (e.g. 1, 3, 5, 10) and types of activations functions (e.g. tanh, linear and 
    ReLU). Always use a linear layer as the output layer! Report your results on the training set as well as the test 
    sets. You can plot your predicted function by passing a list of ascending values for x (e.g. generated with 
    np.linspace) and drawing the resulting curve. Also try and see what happens when you use 2000 neurons with a tanh 
    activation function.
    '''
    # Sigmoid
    nn_model = SkillCraftNN(1, 10, 1, nn.Sigmoid())
    print("\nmodel: %s" % nn_model)
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-1)
    train_nn(1000, nn_model, optimizer, nn_loss_fn, al3_tensor, apm3_tensor, 1, 1)

    nn_model_test_set(al3_tensor, apm3_tensor, nn_model, nn_loss_fn, "NN2 Sigmoid - TEST SET 3")
    nn_model_test_set(al1_tensor, apm1_tensor, nn_model, nn_loss_fn, "NN2 Sigmoid - TEST SET 1")
    nn_model_test_set(al2_tensor, apm2_tensor, nn_model, nn_loss_fn, "NN2 Sigmoid - TEST SET 2")

    # Tanh
    nn_model = SkillCraftNN(1, 10, 1, nn.Tanh())
    print("\nmodel: %s" % nn_model)
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-1)
    train_nn(1000, nn_model, optimizer, nn_loss_fn, al3_tensor, apm3_tensor, 1, 1)

    nn_model_test_set(al3_tensor, apm3_tensor, nn_model, nn_loss_fn, "NN2 Tanh - TEST SET 3")
    nn_model_test_set(al1_tensor, apm1_tensor, nn_model, nn_loss_fn, "NN2 Tanh - TEST SET 1")
    nn_model_test_set(al2_tensor, apm2_tensor, nn_model, nn_loss_fn, "NN2 Tanh - TEST SET 2")

    # ReLU
    nn_model = SkillCraftNN(1, 10, 1, nn.ReLU())
    print("\nmodel: %s" % nn_model)
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-1)
    train_nn(1000, nn_model, optimizer, nn_loss_fn, al3_tensor, apm3_tensor, 1, 1)

    nn_model_test_set(al3_tensor, apm3_tensor, nn_model, nn_loss_fn, "NN2 ReLU - TEST SET 3")
    nn_plot_test_test(al3_tensor, apm3_tensor, nn_model, "NN2 ReLU - TEST SET 3")

    nn_model_test_set(al1_tensor, apm1_tensor, nn_model, nn_loss_fn, "NN2 ReLU - TEST SET 1")
    nn_plot_test_test(al1_tensor, apm1_tensor, nn_model, "NN2 ReLU - TEST SET 1")

    nn_model_test_set(al2_tensor, apm2_tensor, nn_model, nn_loss_fn, "NN2 ReLU - TEST SET 2")
    nn_plot_test_test(al2_tensor, apm2_tensor, nn_model, "NN2 ReLU - TEST SET 2")

    # Tanh 2000 -> SLOW !!! and ineffective
    # 1 - Loss:
    nn_model = SkillCraftNN(1, 2000, 1, nn.Tanh())
    print("\nmodel: %s" % nn_model)
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-1)
    train_nn(1000, nn_model, optimizer, nn_loss_fn, al3_tensor, apm3_tensor, 1, 1)

    nn_model_test_set(al3_tensor, apm3_tensor, nn_model, nn_loss_fn, "NN2 Tanh - TEST SET 3")
    nn_model_test_set(al1_tensor, apm1_tensor, nn_model, nn_loss_fn, "NN2 Tanh - TEST SET 1")
    nn_model_test_set(al2_tensor, apm2_tensor, nn_model, nn_loss_fn, "NN2 Tanh - TEST SET 2")


def lab_2_more_features(col_tensor, apm_tensor, col1_tensor, apm1_tensor, col2_tensor, apm2_tensor, col3_tensor, apm3_tensor):
    """
    Lab2. MORE FEATURES.
    :param col_tensor: 
    :param apm_tensor: 
    :param col1_tensor: 
    :param apm1_tensor: 
    :param col2_tensor: 
    :param apm2_tensor: 
    :param col3_tensor: 
    :param apm3_tensor: 
    :return: notjing
    """
    '''
    ... read all data for APM, ActionLatency, TotalMapExplored, WorkersMade, UniqueUnitsMade, ComplexUnitsMade and 
    ComplexAbilitiesUsed into a tensor. Use the first column as your y tensor, and the other columns as your x tensor, 
    and build a neural network that accepts these 6 inputs and predicts the y-values. Again try several different 
    neural network architectures and report which one produced the best results.
    '''
    nn_model = SkillCraftNN(6, 10, 1, nn.ReLU())
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-1)
    nn_loss_fn = nn.MSELoss()
    print("\nmodel: %s" % nn_model)

    # LAB NOTE: 1000 gives almost the same as 1000
    train_nn(5000, nn_model, optimizer, nn_loss_fn, col3_tensor, apm3_tensor, 6, 1)

    nn_model_y = nn_model(col3_tensor)
    nn_model_loss = loss_fn(nn_model_y, apm3_tensor.view(-1, 1))
    print("\nLOSS for %s: %s " % ("MORE Features - TEST SET 3", nn_model_loss))

    nn_model_y = nn_model(col1_tensor)
    nn_model_loss = loss_fn(nn_model_y, apm1_tensor.view(-1, 1))
    print("\nLOSS for %s: %s " % ("MORE Features - TEST SET 3", nn_model_loss))

    nn_model_y = nn_model(col2_tensor)
    nn_model_loss = loss_fn(nn_model_y, apm2_tensor.view(-1, 1))
    print("\nLOSS for %s: %s " % ("MORE Features - TEST SET 3", nn_model_loss))


def main(filename):
    """
    Main function.
    :param filename: name of the file to read
    """
    '''
    Your first task is to read the data set from the provided CSV file and put the data into tensors.
    We will need two tensors: One for the input (the column ActionLatency) and one for the variable we want to predict (the column APM).
    Implement a function read_csv that takes a file name, and two column names and returns two tensors, one for each of the columns.
    Split the data into three sets: Use the first 30 entries as test set 1, then split the rest randomly into 20% test set 2 and 80% training set.
    '''
    # --- LAB 1 ---
    # read file
    al_tensor, apm_tensor = read_csv_enhanced(filename, ("APM", "ActionLatency"))
    al_tensor = al_tensor.squeeze()
    print("\nal_tensor (%s): %s" % (al_tensor.shape, al_tensor))
    print("apm_tensor (%s): %s" % (apm_tensor.shape, apm_tensor))

    al1_tensor, apm1_tensor, al2_tensor, apm2_tensor, al3_tensor, apm3_tensor = create_test_sets(al_tensor, apm_tensor)
    lab_1(al_tensor, apm_tensor, al1_tensor, apm1_tensor, al2_tensor, apm2_tensor, al3_tensor, apm3_tensor)

    # --- LAB 2 ---
    lab_2(al_tensor, apm_tensor, al1_tensor, apm1_tensor, al2_tensor, apm2_tensor, al3_tensor, apm3_tensor)

    # read file including more features
    columns_tensor, apm_tensor = read_csv_enhanced(filename, ("APM", "ActionLatency", "TotalMapExplored", "WorkersMade",
                                                     "UniqueUnitsMade", "ComplexUnitsMade", "ComplexAbilitiesUsed"))
    print("\ncolumns_tensor (%s): %s" % (columns_tensor.shape, columns_tensor))
    print("apm_tensor (%s): %s" % (apm_tensor.shape, apm_tensor))

    col1_tensor, apm1_tensor, col2_tensor, apm2_tensor, col3_tensor, apm3_tensor = create_test_sets(columns_tensor, apm_tensor)
    lab_2_more_features(columns_tensor, apm_tensor, col1_tensor, apm1_tensor, col2_tensor, apm2_tensor, col3_tensor, apm3_tensor)


if __name__ == "__main__":
    main(sys.argv[1])
