import requests
import tarfile
import os
import numpy as np
from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg import svds
from scipy.optimize import bisect
import time
import matplotlib.pyplot as plt

# Function to download and extract tar.gz file
def download_and_extract(url, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    download_path = os.path.join(save_dir, filename)
    
    response = requests.get(url)
    with open(download_path, "wb") as f:
        f.write(response.content)
    
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(save_dir)

    print(f"Dataset downloaded and extracted successfully to {save_dir}.")

# Function to download a single file
def download_file(url, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    download_path = os.path.join(save_dir, filename)
    
    response = requests.get(url)
    with open(download_path, "wb") as f:
        f.write(response.content)
    
    print(f"File downloaded successfully to {save_dir}.")

# Function to load .fvecs files
def load_fvecs_file(file_path):
    data = []
    with open(file_path, 'rb') as f:
        content = f.read()
        dtype = np.float32
        dim = np.frombuffer(content[:4], dtype=np.int32)[0]
        num_vectors = len(content) // (dim * 4 + 4)
        for i in range(num_vectors):
            vec_start = i * (dim * 4 + 4) + 4
            vec_end = vec_start + dim * 4
            vector = np.frombuffer(content[vec_start:vec_end], dtype=dtype)
            data.append(vector)
    return np.array(data)

# Gradient descent-related functions
def logistic_gradient(W, xi, yi):
    score = np.exp(np.dot(xi, W))
    sum_score = np.sum(score)
    dW = (xi[:, np.newaxis] * score) / sum_score
    dW[:, yi] -= xi
    return dW

def logistic_loss_gradient(W, x, y):
    nu_points = x.shape[0]
    grad = np.zeros_like(W)
    for i in range(nu_points):
        grad += logistic_gradient(W, x[i], y[i])
    return grad / nu_points

def regulizer_subgradient(W, lam):
    U, D, Vt = np.linalg.svd(W, full_matrices=False)
    nuclear_norm = np.sum(D)
    return 2 * lam * nuclear_norm * (U @ Vt)

def logistic_loss(W, x, y):
    nu_points = x.shape[0]
    f = 0
    for i in range(nu_points):
        score = np.exp(np.dot(x[i], W))
        sum_score = np.sum(score)
        f += -np.log(score[y[i]] / sum_score)
    return f / nu_points

def loss_fun(W, x, y, lam):
    _, S, _ = np.linalg.svd(W)
    trace_norm = np.sum(S)
    return logistic_loss(W, x, y) + lam * trace_norm ** 2

def conjugate_min(W, lam):
    _, singular_value, _ = svds(W, k=1)
    return (1 / (2 * lam)) * singular_value * np.outer(_, _)

def finding_root_u(x, lam):
    def fun(root_u):
        return np.sum(np.maximum(((np.sqrt(lam) * x) / root_u - 2 * lam), 0)) - 1

    return bisect(fun, 0, max(x)/2*lam)

def proximal_fun(W, a, lam):
    if np.all(W == 0):
        return W
    U, D, Vt = np.linalg.svd(W, full_matrices=False)
    prox_vec = []
    root_u = finding_root_u(D, lam)
    for d in D:
        l = np.maximum((((np.sqrt(a * lam) * d) / root_u) - 2 * a * lam), 0)
        prox = (l * d) / (l + 2 * a * lam)
        prox_vec.append(prox)
    prox_dig = np.diag(prox_vec)
    return U[:, :10] @ prox_dig @ Vt

# Descent algorithms
def run_subgradient_descent(xi, yi, lam, c, max_itr, initial=None):
    n = xi.shape[1]
    X = initial if initial is not None else np.zeros((n, c))
    min_X, min_value = X, np.inf
    global_count = 1
    x_variables, function_values, function_values_act, itr_time, X_rank = [X], [loss_fun(X, xi, yi, lam)], [loss_fun(X, xi, yi, lam)], [0], [0]
    
    while global_count < max_itr:
        start_time_itr = time.time()

        g = logistic_loss_gradient(X, xi, yi) + regulizer_subgradient(X, lam)
        p = -g / np.linalg.norm(g.T)
        a = 1 / np.sqrt(max_itr)
        X += a * p
        
        temp_value = loss_fun(X, xi, yi, lam)
        if temp_value < min_value:
            min_X, min_value = X, temp_value
        
        rank = np.linalg.matrix_rank(X)
        time_itr = time.time() - start_time_itr

        x_variables.append(min_X)
        function_values.append(min_value)
        function_values_act.append(temp_value)
        X_rank.append(rank)
        itr_time.append(time_itr)

        global_count += 1

    return min_X, min_value, x_variables, function_values, function_values_act, X_rank, itr_time, global_count

def run_gengradient_descent(xi, yi, lam, c, max_itr, initial=None):
    n = xi.shape[1]
    X = initial if initial is not None else np.zeros((n, c))
    min_X, min_value = X, np.inf
    global_count = 1
    x_variables, function_values, function_values_act, itr_time, X_rank = [X], [loss_fun(X, xi, yi, lam)], [loss_fun(X, xi, yi, lam)], [0], [0]

    while global_count < max_itr:
        start_time_itr = time.time()

        g = logistic_loss_gradient(X, xi, yi)
        Y = conjugate_min(-g, lam)
        a = 2 / (global_count + 1)
        X = (1 - a) * X + a * Y

        temp_value = loss_fun(X, xi, yi, lam)
        if temp_value < min_value:
            min_X, min_value = X, temp_value

        rank = np.linalg.matrix_rank(X)
        time_itr = time.time() - start_time_itr

        x_variables.append(min_X)
        function_values.append(min_value)
        function_values_act.append(temp_value)
        X_rank.append(rank)
        itr_time.append(time_itr)

        global_count += 1

    return min_X, min_value, x_variables, function_values, function_values_act, X_rank, itr_time, global_count

def run_proximal(xi, yi, lam, c, max_itr, initial=None):
    n = xi.shape[1]
    X = initial if initial is not None else np.zeros((n, c))
    min_X, min_value = X, np.inf
    global_count = 1
    x_variables, function_values, function_values_act, itr_time, X_rank = [X], [loss_fun(X, xi, yi, lam)], [loss_fun(X, xi, yi, lam)], [0], [0]

    while global_count < max_itr:
        start_time_itr = time.time()

        g = logistic_loss_gradient(X, xi, yi)
        a = 1
        prox = proximal_fun(X - a * g, a, lam)
        G = -(X - prox / a)
        X += a * G

        temp_value = loss_fun(X, xi, yi, lam)
        if temp_value < min_value:
            min_X, min_value = X, temp_value

        rank = np.linalg.matrix_rank(X)
        time_itr = time.time() - start_time_itr

        x_variables.append(X)
        function_values.append(min_value)
        function_values_act.append(temp_value)
        X_rank.append(rank)
        itr_time.append(time_itr)

        global_count += 1

    return min_X, min_value, x_variables, function_values, function_values_act, X_rank, itr_time, global_count

def run_fista(xi, yi, lam, c, max_itr, initial=None):
    n = xi.shape[1]
    X = initial if initial is not None else np.zeros((n, c))
    min_X, min_value = X, np.inf
    X_K1, X_K2 = X, X
    global_count = 2
    x_variables, function_values, function_values_act, itr_time, X_rank = [X, X], [loss_fun(X, xi, yi, lam), loss_fun(X, xi, yi, lam)], [loss_fun(X, xi, yi, lam), loss_fun(X, xi, yi, lam)], [0, 0], [0, 0]

    while global_count < max_itr:
        start_time_itr = time.time()

        g = logistic_loss_gradient(X_K1, xi, yi)
        a = 1
        prox = proximal_fun(X_K1 - a * g, a, lam)
        G = -(X_K1 - prox / a)
        Z_k = X_K1 + a * G

        X_K = (global_count / (global_count + 3)) * (Z_k - X_K1) + Z_k
        temp_value = loss_fun(X_K, xi, yi, lam)

        if temp_value < min_value:
            min_X, min_value = X_K, temp_value

        X_K2, X_K1 = X_K1, X_K

        rank = np.linalg.matrix_rank(X_K1)
        time_itr = time.time() - start_time_itr

        x_variables.append(X_K1)
        function_values.append(min_value)
        function_values_act.append(temp_value)
        X_rank.append(rank)
        itr_time.append(time_itr)

        global_count += 1

    return min_X, min_value, x_variables, function_values, function_values_act, X_rank, itr_time, global_count

# Plotting function
def plot_function_values(iteration, functions, functions_act, function_name):
    plt.plot(iteration, functions, label="Minimized function values")
    plt.plot(iteration, functions_act, label="Actual function values")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(f"{function_name} Descent")
    plt.legend()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Define the URLs and paths
    tgz_url = "http://path/to/tgz/file"
    m_url = "http://path/to/m/file"
    save_dir = "./data"
    tgz_filename = "dataset.tgz"
    m_filename = "data.m"
    
    # Download and extract the files
    download_and_extract(tgz_url, save_dir, tgz_filename)
    download_file(m_url, save_dir, m_filename)
    
    # Load the data
    file_path = os.path.join(save_dir, "your_file.fvecs")
    data = load_fvecs_file(file_path)
    
    # Define parameters and initial values
    xi = np.random.randn(100, 50)
    yi = np.random.randint(0, 2, 100)
    lam = 0.1
    c = 10
    max_itr = 100
    
    # Run different descent methods
    min_X_subgrad, min_value_subgrad, x_vars_subgrad, func_vals_subgrad, func_vals_act_subgrad, X_rank_subgrad, itr_time_subgrad, count_subgrad = run_subgradient_descent(xi, yi, lam, c, max_itr)
    
    min_X_gengrad, min_value_gengrad, x_vars_gengrad, func_vals_gengrad, func_vals_act_gengrad, X_rank_gengrad, itr_time_gengrad, count_gengrad = run_gengradient_descent(xi, yi, lam, c, max_itr)
    
    min_X_proximal, min_value_proximal, x_vars_proximal, func_vals_proximal, func_vals_act_proximal, X_rank_proximal, itr_time_proximal, count_proximal = run_proximal(xi, yi, lam, c, max_itr)
    
    min_X_fista, min_value_fista, x_vars_fista, func_vals_fista, func_vals_act_fista, X_rank_fista, itr_time_fista, count_fista = run_fista(xi, yi, lam, c, max_itr)
    
    # Plot the results
    plot_function_values(range(count_subgrad), func_vals_subgrad, func_vals_act_subgrad, "Subgradient")
    plot_function_values(range(count_gengrad), func_vals_gengrad, func_vals_act_gengrad, "Generalized Gradient")
    plot_function_values(range(count_proximal), func_vals_proximal, func_vals_act_proximal, "Proximal")
    plot_function_values(range(count_fista), func_vals_fista, func_vals_act_fista, "FISTA")
