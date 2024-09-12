# ###################################
# Group ID : 918
# Members : Tanja Bugajski and Nicolai P. B. Pedersen
# Date : 04-09-2024
# Lecture: lecture 2 - 
# Dependencies: 
# Python version: 3.12
# Functionality: 
# ###################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Train data
train_x = np.loadtxt("dataset1_G_noisy_ASCII/trn_x.txt")
train_x_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_x_class.txt")

train_y = np.loadtxt("dataset1_G_noisy_ASCII/trn_y.txt")
train_y_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_y_class.txt")

plt.plot(train_x[:,0],train_x[:,1],".",color="blue")
plt.plot(train_y[:,0],train_y[:,1],".",color="red")
plt.savefig("train_data.png")

# Test data
test_xy = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy.txt")
test_xy_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_class.txt")

plt.figure()
plt.plot(test_xy[:,0],test_xy[:,1],".",color="blue")

test_xy_126 = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126.txt")
test_xy_126_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126_class.txt")

plt.figure()
plt.plot(test_xy_126[:,0],test_xy_126[:,1],".",color="blue")



# Statistics
train_x_mean = np.mean(train_x, axis= 0)
train_x_cov = np.cov(np.transpose(train_x))

train_y_mean = np.mean(train_y, axis= 0)
train_y_cov = np.cov(np.transpose(train_y))

# priors
prior_1_train = len(train_x)/(len(train_x)+len(train_y))
prior_2_train = len(train_y)/(len(train_x)+len(train_y))

# Define likelihood function

def likelihood(data, mean, cov):
     likelihood_value = multivariate_normal.pdf(data,mean,cov)
     return likelihood_value


# To classify the test data we compute the likelihood of it being class x and class y


like_x = likelihood(test_xy,train_x_mean,train_x_cov)
like_y = likelihood(test_xy,train_y_mean,train_y_cov)


# We compute the posterior probability by taking the priors into account


post_x = like_x*prior_1_train
post_y = like_y*prior_2_train


# Now choose to classify our test data as belonging to the class with the highest posterior probability

choice_1 = post_x > post_y


# We can compute the accuracy of our classifications by taking the sum of correct predictions and divide by the total number of predictions

correct_guesses_1 = choice_1 == (test_xy_label-2)*-1

accuracy_xy = np.sum(correct_guesses_1)/len(test_xy)


# ### (b) classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses, and use the corresponding label file tst_xy_126_class to calculate the accuracy;

# First we define our prior probabilities


prior_x_uniform = 0.5
prior_y_uniform = 0.5


# We can now compute posteriors knowing that the posterior probability is simply the prior, p(C), multiplied by the likelihood p(x, C).



likelihood_x_uniform = likelihood(test_xy_126,train_x_mean,train_x_cov)
likelihood_y_uniform = likelihood(test_xy_126,train_y_mean,train_y_cov)

posterior_x_uniform = likelihood_x_uniform
posterior_y_uniform = likelihood_y_uniform



classification_uniform = posterior_x_uniform > posterior_y_uniform

correct_guesses_2 = (test_xy_126_label-2)*-1 == classification_uniform

accuracy_xy_126_uniform = np.sum(correct_guesses_2)/len(test_xy_126)

print(f"Accuracy using uniform prior {accuracy_xy_126_uniform*100:.2f}%")


# ### (c) classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).

# Here we simply follow the procedure of (b), however, this time with updated priors


prior_x_non_uniform = 0.9
prior_y_non_uniform = 0.1

likelihood_x_non_uniform = likelihood_x_uniform
likelihood_y_non_uniform = likelihood_y_uniform

posterior_x_non_uniform = likelihood_x_uniform*prior_x_non_uniform
posterior_y_non_uniform = likelihood_y_uniform*prior_y_non_uniform

classification_non_uniform = posterior_x_non_uniform > posterior_y_non_uniform

correct_guesses_3 = (test_xy_126_label-2)*-1 == classification_non_uniform

accuracy_xy_126_non_uniform = np.sum(correct_guesses_3)/len(test_xy_126)

print(f"Accuracy using non-uniform prior {accuracy_xy_126_non_uniform*100:.2f}%")


# Comparing the accuracy using uniform prior and non-uniform priors we see that using prior information about the data distribution improves classifcation accuracy by ?%.


improvement = (accuracy_xy_126_non_uniform / accuracy_xy_126_uniform) - 1
print(f"Absolute improvement in accuracy {improvement*100:.2f}%")





