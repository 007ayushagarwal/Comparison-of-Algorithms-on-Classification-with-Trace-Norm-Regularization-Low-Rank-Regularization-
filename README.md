# Comparasinon-of-different-optmization-algorithm
Comparison of different algorithms on classification with trace norm regularisation(low rank regularisation)


## Overview
This project implements various optimization algorithms used in convex optimization problems, including Subgradient Descent, Generalized Gradient Descent, Proximal Gradient Descent, and Fast Iterative Shrinkage-Thresholding Algorithm (FISTA). The code is designed to work with datasets stored in `.fvecs` format, and includes functions for downloading, extracting, and processing data.

## Dependencies

- Python 3.6+
- `requests`
- `numpy`
- `scipy`
- `matplotlib`

If you don't have these dependencies installed, you can install them using `pip`:

```bash
pip install requests numpy scipy matplotlib
```

Or using `conda`:

```bash
conda install requests numpy scipy matplotlib
```

## Project Structure

- `convexoptimizationproject.py`: The main script containing all the functions for downloading datasets, loading `.fvecs` files, and running various optimization algorithms.
- `README.md`: This file, providing an overview and instructions on how to use the code.

## Functions

### Data Handling

- **`download_and_extract(url, save_dir, filename)`**: Downloads a tar.gz file from a specified URL and extracts its contents to a given directory.
  
- **`download_file(url, save_dir, filename)`**: Downloads a single file from a specified URL and saves it to a given directory.
  
- **`load_fvecs_file(file_path)`**: Loads data from a `.fvecs` file and returns it as a NumPy array.

### Optimization Algorithms

- **`logistic_gradient(W, xi, yi)`**: Computes the gradient of the logistic loss function.
  
- **`logistic_loss_gradient(W, x, y)`**: Computes the gradient of the logistic loss for a batch of data points.
  
- **`regulizer_subgradient(W, lam)`**: Computes the subgradient of the regularization term based on the nuclear norm.
  
- **`logistic_loss(W, x, y)`**: Computes the logistic loss for a set of weights and data points.
  
- **`loss_fun(W, x, y, lam)`**: Computes the total loss, combining logistic loss and regularization.
  
- **`conjugate_min(W, lam)`**: Finds the conjugate minimum for generalized gradient descent.
  
- **`finding_root_u(x, lam)`**: Helper function to find the root for the proximal function.
  
- **`proximal_fun(W, a, lam)`**: Implements the proximal operator.
  
- **`run_subgradient_descent(xi, yi, lam, c, max_itr, initial=None)`**: Runs the Subgradient Descent algorithm.
  
- **`run_gengradient_descent(xi, yi, lam, c, max_itr, initial=None)`**: Runs the Generalized Gradient Descent algorithm.
  
- **`run_proximal(xi, yi, lam, c, max_itr, initial=None)`**: Runs the Proximal Gradient Descent algorithm.
  
- **`run_fista(xi, yi, lam, c, max_itr, initial=None)`**: Runs the FISTA algorithm.

### Plotting

- **`plot_function_values(iteration, functions, functions_act, function_name)`**: Plots the function values over iterations for different descent methods.

## Usage

### Step 1: Set up URLs and Paths

Edit the URLs and file paths in the `__main__` section of the script to point to your dataset.

### Step 2: Run the Script

Run the script with:

```bash
python convexoptimizationproject.py
```

This will download the dataset, run the different optimization algorithms, and plot the results.

### Step 3: Analyze Results

The script generates plots comparing the minimized function values and actual function values over iterations for each descent method. You can use these plots to analyze the performance of each optimization technique.

## Example

```python
if __name__ == "__main__":
    tgz_url = "http://path/to/tgz/file"
    m_url = "http://path/to/m/file"
    save_dir = "./data"
    tgz_filename = "dataset.tgz"
    m_filename = "data.m"
    
    download_and_extract(tgz_url, save_dir, tgz_filename)
    download_file(m_url, save_dir, m_filename)
    
    file_path = os.path.join(save_dir, "your_file.fvecs")
    data = load_fvecs_file(file_path)
    
    xi = np.random.randn(100, 50)
    yi = np.random.randint(0, 2, 100)
    lam = 0.1
    c = 10
    max_itr = 100
    
    min_X_subgrad, min_value_subgrad, x_vars_subgrad, func_vals_subgrad, func_vals_act_subgrad, X_rank_subgrad, itr_time_subgrad, count_subgrad = run_subgradient_descent(xi, yi, lam, c, max_itr)
    
    min_X_gengrad, min_value_gengrad, x_vars_gengrad, func_vals_gengrad, func_vals_act_gengrad, X_rank_gengrad, itr_time_gengrad, count_gengrad = run_gengradient_descent(xi, yi, lam, c, max_itr)
    
    min_X_proximal, min_value_proximal, x_vars_proximal, func_vals_proximal, func_vals_act_proximal, X_rank_proximal, itr_time_proximal, count_proximal = run_proximal(xi, yi, lam, c, max_itr)
    
    min_X_fista, min_value_fista, x_vars_fista, func_vals_fista, func_vals_act_fista, X_rank_fista, itr_time_fista, count_fista = run_fista(xi, yi, lam, c, max_itr)
    
    plot_function_values(range(count_subgrad), func_vals_subgrad, func_vals_act_subgrad, "Subgradient")
    plot_function_values(range(count_gengrad), func_vals_gengrad, func_vals_act_gengrad, "Generalized Gradient")
    plot_function_values(range(count_proximal), func_vals_proximal, func_vals_act_proximal, "Proximal")
    plot_function_values(range(count_fista), func_vals_fista, func_vals_act_fista, "FISTA")
```

