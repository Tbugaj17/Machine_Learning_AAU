# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# %%
# Load data
work_dir = os.getcwd() # CHANGE THIS WITH YOUR CURRENT DIR!
save_path = work_dir + r'\ex01_data.npy'

data = np.load("ex01_data.npy") # TODO: load data (check https://numpy.org/doc/stable/reference/generated/numpy.load.html)
    
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]

# %%
# Sample data and plot
np.random.seed(100)

sample_points = 10
sample_idx = np.sort(np.random.choice(len(x1), sample_points))

x1_sub = np.take(x1, sample_idx)
x2_sub = np.take(x2, sample_idx) # TODO: sample x2

plt.figure()
plt.scatter(x1_sub, x2_sub, label='Sampled data')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()

# %%
def polyfit(x1, x2, deg, regularization=0, y=None, show_sums_of_squares=False):
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=deg)
    x1_poly = poly.fit_transform(x1[:, np.newaxis])
    
    # Create the Ridge regression model
    ridge_reg = Ridge(alpha=regularization)
    
    # Fit the model (reshape x1_poly to be 2D if needed)
    ridge_reg.fit(x1_poly, x2)
    
    # Generate x values for the regression line/curve
    x = np.linspace(np.min(x1), np.max(x1), 1000)  # Generate 1000 samples between the minimum and maximum values of x1
    X_poly = poly.transform(x[:, np.newaxis])
    
    # Predict y_hat values
    y_hat = ridge_reg.predict(X_poly)  # Use the ridge regression to predict x2 given x1
    
    # Extract coefficients
    coefs = (ridge_reg.intercept_, *ridge_reg.coef_[1:])
    
    # Calculate Mean Squared Error (MSE)
    MSE = (1/len(x1)) * np.sum((ridge_reg.predict(x1_poly) - x2)**2)  # Implement MSE equation
    
    # Plot observations and the regression line/curve
    plt.figure()
    
    # Show the "squares" to be minimized
    if show_sums_of_squares:
        plt.plot((x1, x1), (x2, ridge_reg.predict(x1_poly)), color='limegreen')
    
    plt.scatter(x1, x2, label='Observations')
    plt.plot(x, y_hat, '-', label='Polynomial fit of degree %i' % deg, color='orange')
    
    if y is not None:
        plt.plot(np.linspace(-5, 5, 500), y, label='Ground truth', color='red')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim([-20, 40])
    plt.legend()
    plt.grid()
    plt.show()
    
    return coefs, MSE


# %%
# 1) Experiment with different polynomial degrees:
for deg in [0,1,2,3,4,9]:

    coefs, mse = polyfit(x1_sub, x2_sub, deg=deg, y=y, show_sums_of_squares=True)
    print('MSE = ', round(mse,3))

"""
For the first experiment, we see an increase in the polynomial degree yields a greater fit, this is also proven by the MSE
which decreases as the degree increases
"""

# %%
# 2) Explore L2 regularization:
for reg in [0.01,0.001,0.0001]:
    deg = 9

    coefs, mse = polyfit(x1_sub,x2_sub,deg=deg,regularization=reg,y=y) # TODO
    print(f"degree = {deg}, regularization = {reg}")
    print('MSE = ', round(mse,3))

"""
When the regularization parameter is increased, the polynomial appears smoother, though the MSE increases. 
A prediction will thus be better for a higher regularization parameter, though the observations are better fitted with
a lower regularization parameter.
"""

# %%
# 3) Investigate the effect of sample size.
np.random.seed(100)

sample_points = 100
sample_idx = np.sort(np.random.choice(len(x1), sample_points)) # TODO

x1_sub_100 = np.take(x1, sample_idx) # TODO
x2_sub_100 = np.take(x2, sample_idx) # TODO


deg = 9
reg = 1e-2
coefs, mse = polyfit(x1_sub_100, x2_sub_100, deg=deg, regularization=reg, y=y)
print('MSE = ', round(mse,3))

"""
We see the fluctuation of the polynomial is decreased, hence the overfitting has been solved. 
"""
