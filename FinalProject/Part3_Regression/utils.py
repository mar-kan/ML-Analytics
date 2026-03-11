import matplotlib.pyplot as plt
import numpy as np


def print_list_MSE(method, results, keyword):
    print(f"{method} MSE Results:")
    method_results = {k: v for k, v in results.items() if keyword in k}

    for model, mse in method_results.items():
        print(f"  - {model}: {mse:.4f}")

def plot_regression_results_multi_p(method, n_plot, y_test_f, ps, X_test_mat, betas):
    plot_indices = np.arange(n_plot)
    y_subset = y_test_f[:n_plot].flatten()

    plt.figure(figsize=(12, 6))

    plt.plot(plot_indices, y_subset, 'k.', markersize=10, label='Actual Berry Size', alpha=0.7)
    for i, p in enumerate(ps):
        y_hat_p = (X_test_mat[p][:n_plot] @ betas[p]).flatten()    
        plt.plot(plot_indices, y_hat_p, label=f'{method} Degree p={p}')

    plt.xlabel('Berry Index')
    plt.ylabel('Berry Size')
    plt.title(f'{method} Regression', fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_regression_results_matrix(method, n_plot, y_test_f, p, X_test_mat, betas):
    plot_indices = np.arange(n_plot)
    y_subset = y_test_f[:n_plot].flatten()

    plt.figure()
    plt.plot(plot_indices, y_subset, 'k.', markersize=10, label='Actual Berry Size', alpha=0.7)
    y_hat_full = (X_test_mat[p] @ betas).flatten()
    y_hat_subset = y_hat_full[:n_plot]

    plt.plot(plot_indices, y_hat_subset, label=f'Standard Poly p={p}')

    plt.xlabel('Berry Index')
    plt.ylabel('Berry Size')
    plt.title(f'{method} Regression p=1', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_general_regression_results(method, n_plot, y_test_f, y_hat):
    plot_indices = np.arange(n_plot)
    y_subset = y_test_f[:n_plot].flatten()
    y_hat_subset = y_hat[:n_plot]

    plt.figure()
    plt.plot(plot_indices, y_subset, 'k.', markersize=10, label='Actual Berry Size', alpha=0.7)
    plt.plot(plot_indices, y_hat_subset, label=f"{method} Regression")

    plt.xlabel('Berry Index')
    plt.ylabel('Berry Size')
    plt.title(f'{method} Regression', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
