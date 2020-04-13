import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


def read_csv(filename, col_x, col_y):
    """
    Read CSV file.

    :param filename: name of the file to read
    :param col_x: name of column X
    :param col_y: name of column Y
    :return: a tuple with two tensors, one per column
    """
    data = pd.read_csv(filename)
    x = torch.tensor(data[col_x])
    y = torch.tensor(data[col_y])
    return x, y


def ols(tensor_x, tensor_y):
    """
    Calculate Ordinary Least Squares for function like y=wx + b.

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


def scatterplot_test_set(al_tensor, apm_tensor, name):
    """
    Scatterplot a given test set.

    :param al_tensor: tensor X
    :param apm_tensor: tensor y
    :param name: figure name
    """
    plt.figure(num=name)
    plt.scatter(al_tensor, apm_tensor)
    plt.xlabel("Action Latency")
    plt.ylabel("APM")
    plt.show()


def plot_test_set_ols(al_tensor, apm_tensor, ols_coefficients, name):
    """
    Plot a given test set.

    :param al_tensor: tensor X
    :param apm_tensor: tensor y
    :param ols_coefficients: OLS coefficients
    :param name: figure name
    """
    apm_tensor_ols = al_tensor * ols_coefficients[0] + ols_coefficients[1]
    print("\nplot_test_set_ols")
    print("al_tensor (%s): %s..." % (al_tensor.shape[0], al_tensor[:5]))
    print("apm_tensor (%s): %s..." % (apm_tensor.shape[0], apm_tensor[:5]))
    print("apm_tensor_osl (%s): %s..." % (apm_tensor_ols.shape[0], apm_tensor_ols[:5]))

    plt.figure(num=name)
    plt.scatter(al_tensor, apm_tensor)
    plt.plot(al_tensor, apm_tensor_ols)
    plt.scatter(al_tensor, apm_tensor_ols)
    plt.xlabel("Action Latency")
    plt.ylabel("APM")
    plt.show()


def model(tensor_w, tensor_x, tensor_b):
    """
    Accepts three tensors, w, x and b and returns the value for y where y = wx + b.

    :param tensor_w: w tensor
    :param tensor_x: x tensor
    :param tensor_b: v tensor
    :return: the value for y where y = wx + b
    """
    y = tensor_w * tensor_x
    #print("y (%s): %s" % (y.shape, y))
    y = y + tensor_b
    #print("y (%s): %s" % (y.shape, y))
    return y


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
    # or alternatively def dmodel_w(tensor_x, tensor_w, tensor_b):
    return tensor_x


def dmodel_b():
    # or alternatively def dmodel_b(tensor_x, tensor_w, tensor_b):
    return 1


def dloss_m(tensor_y, tensor_real_y):
    return -2 * (tensor_real_y - tensor_y)


def training(iterations, tensor_w, tensor_b, alpha, tensor_x, tensor_y):
    print("\n*** TRAINING ***")

    for it in range(iterations):
        print("\n--- Iteration %s" % it)
        tensor_y_calc = model(tensor_w, tensor_x, tensor_b)
        loss = loss_fn(tensor_y_calc, tensor_y)
        print("LOSS: %s" % loss)

        gradient_w = (dloss_m(tensor_y_calc, tensor_y) * dmodel_w(tensor_x)).mean()
        gradient_b = (dloss_m(tensor_y_calc, tensor_y) * dmodel_b()).mean()
        print("gradient_w: %s - gradient_b: %s" % (gradient_w, gradient_b))

        tensor_w = tensor_w - alpha * gradient_w
        tensor_b = tensor_b - alpha * gradient_b
        print("tensor_w: %s - tensor_b: %s " % (tensor_w, tensor_b))

    return tensor_w, tensor_b


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
    al_tensor, apm_tensor = read_csv(filename, "ActionLatency", "APM")

    # build Set 1 with the first 30 entries (TEST SET 1)
    al1_tensor = al_tensor[:30]
    apm1_tensor = apm_tensor[:30]
    print("\nal1_tensor (%s): %s" % (al1_tensor.shape, al1_tensor))
    print("apm1_tensor (%s): %s" % (apm1_tensor.shape, apm1_tensor))

    # build an intermediate set with all remaining records beyond index 30
    remaining_al_tensor = al_tensor[30:]
    remaining_apm_tensor = apm_tensor[30:]
    # print("remaining_al_tensor (%s): %s" % (remaining_al_tensor.shape, remaining_al_tensor))
    # print("remaining_apm_tensor (%s): %s" % (remaining_apm_tensor.shape, remaining_apm_tensor))

    # calculate the size of the remaining data  and the index where its first 20% ends
    remaining_size = remaining_al_tensor.shape[0]
    twenty_percent_index = (remaining_size * 20) // 100
    print("\nremaining_size: %s" % remaining_size)
    print("twenty_percent_index: %s" % twenty_percent_index)

    # create random permutation if integers from 0 to remaining_size
    random_perm = torch.randperm(remaining_size)
    print("random_perm (%s): %s" % (random_perm.shape, random_perm))

    # build Set 2 with 20% of remaining data (TEST SET 2)
    al2_tensor = remaining_al_tensor[random_perm[:twenty_percent_index]]
    apm2_tensor = remaining_apm_tensor[random_perm[:twenty_percent_index]]
    print("\nal2_tensor shape: %s" % al2_tensor.shape)
    print("apm2_tensor shape: %s" % apm2_tensor.shape)

    # build Set 3 with 80% of remaining data (TRAINING SET)
    al3_tensor = remaining_al_tensor[random_perm[twenty_percent_index:]]
    apm3_tensor = remaining_apm_tensor[random_perm[twenty_percent_index:]]
    print("\nal3_tensor (%s): %s" % (al3_tensor.shape, al3_tensor))
    print("apm3_tensor (%s): %s" % (apm3_tensor.shape, apm3_tensor))

    '''
    Exploratory Data Analysis
    Take the test set 1 (with thirty entries), and plot it as a scatterplot with Matplotlib. 
    Does the data look linear? Try the same with the training set. 
    What are the maximum, minimum and mean values for APM and ActionLatency? 
    What is the standard deviation of the two variables? 
    What is the correlation between the two variables?
    '''
    # scatter plot and line in the same figure
    #scatterplot_test_set(al1_tensor, apm1_tensor, "Test Set 1")
    #scatterplot_test_set(al3_tensor, apm3_tensor, "Test Set 3/Training Set")

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

    #plot_test_set_ols(al1_tensor, apm1_tensor, ols_result, "Test Set 1")
    #plot_test_set_ols(al2_tensor, apm2_tensor, ols_result, "Test Set 2")
    #plot_test_set_ols(al3_tensor, apm3_tensor, ols_result, "Test Set 3/Training Set")

    tensor_w = torch.empty(1)
    tensor_b = torch.empty(1)
    tensor_w[0] = -2
    tensor_b[0] = 230

    # TRAINING - NOT NORMALIZED
    #tensor_w, tensor_b = training(1000, tensor_w, tensor_b, 1e-4, al3_tensor, apm3_tensor)
    #tensor_w, tensor_b = training(10000, tensor_w, tensor_b, 1e-4, al3_tensor, apm3_tensor)

    # TRAINING - NORMALIZED
    print("\nal3tensor: %s" % al3_tensor)
    print("al3tensor mean: %s - max: %s - min %s" % (torch.mean(al3_tensor), torch.max(al3_tensor), torch.min(al3_tensor)))
    al3_tensorn = al3_tensor - torch.mean(al3_tensor)
    print("al3tensorn: %s" % al3_tensorn)
    print("al3tensorn DEN: %s" % ((torch.max(al3_tensor) - torch.min(al3_tensor)) / 2))
    al3_tensorn = al3_tensorn / ((torch.max(al3_tensor) - torch.min(al3_tensor)) / 2)
    print("al3tensorn: %s" % al3_tensorn)

    tensor_w, tensor_b = training(10000, tensor_w, tensor_b, 1e-1, al3_tensorn, apm3_tensor)

    # TRAINING - PLOT
    #plot_test_set_ols(al3_tensorn, apm3_tensor, (tensor_w, tensor_b), "NEW")

    # scikit-learn result
    from sklearn.linear_model import LinearRegression
    linr = LinearRegression()
    linr.fit(al3_tensor.reshape(-1, 1), apm3_tensor)
    print("\nsklearn.linear_model: %s, %s" % (linr.coef_[0], linr.intercept_))


if __name__ == "__main__":
    main(sys.argv[1])
