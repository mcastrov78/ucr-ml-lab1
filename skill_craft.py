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
    print("\nx (%s): %s" % (tensor_x.shape, tensor_x))
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
    print("\nal_tensor (%s): %s..." % (al_tensor.shape[0], al_tensor[:5]))
    print("apm_tensor (%s): %s..." % (apm_tensor.shape[0], apm_tensor[:5]))
    print("apm_tensor_osl (%s): %s..." % (apm_tensor_ols.shape[0], apm_tensor_ols[:5]))

    plt.figure(num=name)
    plt.scatter(al_tensor, apm_tensor)
    plt.plot(al_tensor, apm_tensor_ols)
    plt.scatter(al_tensor, apm_tensor_ols)
    plt.xlabel("Action Latency")
    plt.ylabel("APM")
    plt.show()


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
    scatterplot_test_set(al1_tensor, apm1_tensor, "Test Set 1")
    scatterplot_test_set(al3_tensor, apm3_tensor, "Test Set 3/Training Set")

    # print some statistics
    print("")
    print("max AL: %s - APM: %s" % (torch.max(al_tensor), torch.max(apm_tensor)))
    print("min AL: %s - APM: %s" % (torch.min(al_tensor), torch.min(apm_tensor)))
    print("mean AL: %s - APM: %s" % (torch.mean(al_tensor), torch.mean(apm_tensor)))
    print("std AL: %s - APM: %s" % (torch.std(al_tensor), torch.std(apm_tensor)))
    print("corr AL-APM: %s" % (np.corrcoef(al_tensor, apm_tensor)))

    # calculate Ordinary Least Squares from the Training Set (#3)
    ols_result = ols(al3_tensor, apm3_tensor)
    print("\nOSL: %s" % ols_result)

    plot_test_set_ols(al1_tensor, apm1_tensor, ols_result, "Test Set 1")
    plot_test_set_ols(al2_tensor, apm2_tensor, ols_result, "Test Set 2")
    plot_test_set_ols(al3_tensor, apm3_tensor, ols_result, "Test Set 3/Training Set")


if __name__ == "__main__":
    main(sys.argv[1])
