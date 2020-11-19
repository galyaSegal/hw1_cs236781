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
$\Delta$ is a hyperparameter of the model, which allows some error for wrong predictions (margin). Since it is a 
hyperparameter, it can be tuned and its value is arbitrary as long as it is positive (sustains the 'direction' of the 
error).

"""

part3_q2 = r"""
**Your answer:**
1. The linear model tries to fit non-linear shapes (numbers) by multiple linear classifications. For example, in the 
first error the model classified a 5 digit as a 6 digit. If we look closely in the written digit, it is not a 'typical'
 5 digit, and it has multiple lines which resemble the shape of a 5 digit.
 
2. A KNN model can be thought of as a classification which finds the digit which resembles most the one we are 
 classifying. It is somewhat different from linear classification, which fits linear lines to the non-linear shapes. 
 However, both models try to learn characteristics which are unique for each digit.

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
**Your answer:**

"""

part4_q2 = r"""
**Your answer:**


```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
