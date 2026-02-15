""" hereditary_traits.py """
import numpy as np

def generate_data(mean, var, hrd, n_each, n):
   g = mean + np.sqrt(hrd*var)*np.random.randn(n,1) # shared genetic factor
   gs = np.tile(g, (1, n_each))
   x = gs + np.sqrt((1-hrd)*var)*np.random.randn(n,n_each) # offspring heights
   return g, x


def generate_data_diff_means(mean_parent, mean_offspring, var, hrd, n_each, n):
   g = np.sqrt(hrd*var)*np.random.randn(n,1) # shared genetic factor (parent)
   gs = np.tile(g, (1, n_each))

   p = gs + mean_parent + np.sqrt((1-hrd)*var)*np.random.randn(n,n_each)
   x = gs + mean_offspring + np.sqrt((1-hrd)*var)*np.random.randn(n,n_each) # offspring heights
   return g, p, x
