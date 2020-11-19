r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. It allows estimation of the performance of the model on unseen data, so we can estimate how well it generalized.
2. We will select a subset of the data which best represents the dataset.
3. The test set should be used **only** when evaluating the performance of the model. 
The validation set is used for cross-validation and for the decision of the best model to be used.
"""

part1_q2 = r"""
**Your answer:**
Yes, we need to set aside some data for validation in order to tune hyperparameters and choose the best model. 
Using the test set for that purpose means overfitting on the test set ('cheating'). 
In such case we can't evaluate how well the model generalizes.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
No, it depends on the data. For example, if $$k >= Number of observations$$, all our predictions will end up as the 
majority class, and thus usually the bias will be high and the variance will be low, which end (usually) in worse 
generalization.
"""

part2_q2 = r"""
**Your answer:**
1. Choosing the best model with respect to the train set, which we trained on, will result in overfitting on the
 train set. We will maximize the score over the train set without evaluating the model generalization capabilities. 
 In such case, the performance on the test set are expected to be poor.
2. Choosing the model with the best performance on the test set will result in overfitting on the test set. The test
set should be used only once for evaluation, not for selection of the best model.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
$\Delta$ is a hyperparameter of the model, which allows some error for wrong predictions (margin). We also added
regularization on the weights, and the weights can be thought of as the distance between the sample and the decision 
boundary. Changing $\Delta$ will result in a different model with different weights, so the value of $\Delta$ is
arbitrary.

"""

part3_q2 = r"""
**Your answer:**
1. The linear model learns the locations of each label and then fits the digit which resembles the cuurent sample 
 the most. For example, in the first error the model classified a 5 digit as a 6 digit. If we look closely in the
 written digit, it is not a 'typical' 5 digit- it resembles the shape of a 5 digit, and it is quite clear why the model
 wrongly classified this specific sample.
 
2. A KNN model can be thought of as a classification which finds the digit which uses the k closest samples with the
 features (locations) which resembles most the one we are currently classifying. It is somewhat different from linear
classification, which uses all the data to learn the locations of each label. 
 However, both models try to learn the locations which are unique for each digit.

"""

part3_q3 = r"""
**Your answer:**

1. The learning rate we chose is possibly too high because the model converges fast and has a sudden big error after 
the convergence. This can be explained as a big step taken which is overshooting the minimum. A learning rate too low
would cause a slow convergence and would reach lower accuracy in the same number of epochs.

2. The model is slightly overfitting the train set sinch there is a ~7% difference in the accuracies of the train and 
the test set. It reaches ~88% accuracy on the test set, so it is not highly overfitted.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern will be with the residuals around and as close to zero with a fixed variance over y_pred. <br>
<br>

It seems the residuals on the final plot are closer to zero and more uniformly spread than the first top-5 plot which gets a kind of curved shape in response to y_pred. <br>
Thus, we think the final model fits the data well and better than the previous model. <br>

"""

part4_q2 = r"""

1. Yes, as the linearity is considered relatively to $\mat{W}$ and not relatively to the original features. <br>
2. Yes, as any non-linear relation between the original features can be obtained by feature engineering at the price of higher dimension. <br>
3. The decision boundary after adding the non-linear features will be of higher space. It would still be a hyperplane as the model of linear regression always learn a linear separator (as it is still linear in terms of $\mat{W}$). <br>
"""

part4_q3 = r"""

1. We used np.logspace instead of np.linspace as logarithmic scale enables us to search a bigger space with dramatically different values quickly. <br> 
In addition, it enables us to be more sensitive regard smaller values as we expect less variation in the results between bigger values of the regularization coefficient.<br>

2. We fitted the model k_folds times for each combination of the hyperparameters (which delved of the given ranges)  <br>

<br>
|degree_range = np.arange(1, 4)| = 3 <br>
|lambda_range = np.logspace(-3, 2, base=10, num=20)| = 20<br>
k_folds = 3<br>
<br>
Total_fitted = 3 * 20 *3  = 180 times

"""

# ==============
