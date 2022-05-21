# Train faster, more flexible models with Amazon SageMaker Linear Learner

Amazon SageMaker has launched several additional features to the built-in linear learner algorithm.  Amazon SageMaker algorithms are designed to scale effortlessly to massive datasets and take advantage of the latest hardware optimizations for unparalleled speed.  The Amazon SageMaker linear learner algorithm encompasses both linear regression and binary classification algorithms.  These algorithms are used extensively in banking, fraud/risk management, insurance, and healthcare.  The new features of linear learner are designed to speed up training and help you customize models for different use cases.  Examples include classification with unbalanced classes, where one of your outcomes happens far less frequently than another.  Or specialized loss functions for regression, where it’s more important to penalize certain model errors more than others.

In this series of notebooks, we'll cover two features that can be easily implemented using SageMaker Linear Learner:
1. Early stopping and saving the best model
1. New ways to customize linear learner models, including:
   * Hinge loss (support vector machines)
   * Quantile loss
   * Huber loss
   * Epsilon-insensitive loss
   * Class weights options

In our notebook series, we'll walk you through a hands-on example of using class weights to boost performance in binary classification using these features.


## Early Stopping

Linear learner trains models using Stochastic Gradient Descent (SGD) or variants of SGD like Adam.  Training requires multiple passes over the data, called *epochs*, in which the data are loaded into memory in chunks called *batches*, sometimes called *minibatches*.  How do we know how many epochs to run?  Ideally, we'd like to continue training until convergence - that is, until we no longer see any additional benefits.  Running additional epochs after the model has converged is a waste of time and money, but guessing the right number of epochs is difficult to do before submitting a training job.  If we train for too few epochs, our model will be less accurate than it should be, but if we train for too many epochs, we'll waste resources and potentially harm model accuracy by overfitting.  To remove the guesswork and optimize model training, linear learner has added two new features: automatic early stopping and saving the best model.  

Early stopping works in two basic regimes: with or without a validation set.  Often we split our data into training, validation, and testing data sets.  Training is for optimizing the loss, validation is for tuning hyperparameters, and testing is for producing an honest estimate of how the model will perform on unseen data in the future.  If you provide linear learner with a validation data set, training will stop early when validation loss stops improving.  If no validation set is available, training will stop early when training loss stops improving.

#### Early Stopping with a validation data set
One big benefit of having a validation data set is that we can tell if and when we start overfitting to the training data.  Overfitting is when the model gives predictions that are too closely tailored to the training data, so that generalization performance (performance on future unseen data) will be poor.  The following plot on the right shows a typical progression during training with a validation data set.  Until epoch 5, the model has been learning from the training set and doing better and better on the validation set.  But in epochs 7-10, we see that the model has begun to overfit on the training set, which shows up as worse performance on the validation set.  Regardless of whether the model continues to improve (overfit) on the training data, we want to stop training after the model starts to overfit.  And we want to restore the best model from just before the overfitting started.  These two features are now turned on by default in linear learner.  

The default parameter values for early stopping are shown in the following code.  To tweak the behavior of early stopping, try changing the values.  To turn off early stopping entirely, choose a patience value larger than the number of epochs you want to run.


    early_stopping_patience=3,
    early_stopping_tolerance=0.001,
    
The parameter early_stoping_patience defines how many epochs to wait before ending training if no improvement is made.  It's useful to have a little patience when deciding to stop early, since the training curve can be bumpy.  Performance may get worse for one or two epochs before continuing to improve.  By default, linear learner will stop early if performance has degraded for three epochs in a row.

The parameter early_stopping_tolerance defines the size of an improvement that's considered significant.  If the ratio of the improvement in loss divided by the previous best loss is smaller than this value, early stopping will consider the improvement to be zero.

#### Early stopping without a validation data set

When training with a training set only, we have no way to detect overfitting.  But we still want to stop training once the model has converged and improvement has levelled off.  In the left panel of the following figure, that happens around epoch 25.

<img src="images/early_stop.png">

#### Early stopping and calibration
You may already be familiar with the linear learner automated threshold tuning for binary classification models.  Threshold tuning and early stopping work together seamlessly by default in linear learner.  

When a binary classification model outputs a probability (e.g., logistic regression) or a raw score (SVM), we convert that to a binary prediction by applying a threshold, for example:

    predicted_label = 1 if raw_prediction > 0.5 else 0
    
We might want to tune the threshold (0.5 in the example) based on the metric we care about most, such as accuracy or recall.  Linear learner does this tuning automatically using the 'binary_classifier_model_selection_criteria' parameter.  When threshold tuning and early stopping are both turned on (the default), then training stops early based on the metric you request.  For example, if you provide a validation data set and request a logistic regression model with threshold tuning based on accuracy, then training will stop when the model with auto-thresholding reaches optimal performance on the validation data.  If there is no validation set and auto-thresholding is turned off, then training will stop when the best value of the loss function on the training data is reached.


## New loss functions

The loss function is our definition of the cost of making an error in prediction.  When we train a model, we push the model weights in the direction that minimizes loss, given the known labels in the training set.  The most common and well-known loss function is squared loss, which is minimized when we train a standard linear regression model.  Another common loss function is the one used in logistic regression, variously known as logistic loss, cross-entropy loss, or binomial likelihood.  Ideally, the loss function we train on should be a close match to the business problem we're trying to solve.  Having the flexibility to choose different loss functions at training time allows us to customize models to different use cases.  In this section, we'll discuss when to use which loss function, and introduce several new loss functions that have been added to linear learner.

<img src="images/loss_functions.png">

### Squared loss

    predictor_type='regressor',
    loss='squared_loss',

$$\text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^{N} (w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i)^2$$

We'll use the following notation in all of the loss functions we discuss:

$w_0$ is the bias that the model learns

$\mathbf{w}$ is the vector of feature weights that the model learns

$y_i$ and $\mathbf{x_i}$ are the label and feature vector, respectively, from example $i$ of the training data

$N$ is the total number of training examples

Squared loss is a first choice for most regression problems.  It has the nice property of producing an estimate of the mean of the label given the features.  As seen in the plot above, squared loss implies that we pay a very high cost for very wrong predictions.  This can cause problems if our training data include some extreme outliers.  A model trained on squared loss will be very sensitive to outliers.  Squared loss is sometimes known as mean squared error (MSE), ordinary least squares (OLS), or $\text{L}_2$ loss.   Read more about [squared loss](https://en.wikipedia.org/wiki/Least_squares) on wikipedia.

### Absolute loss

    predictor_type='regressor',
    loss='absolute_loss',
    
$$\text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^{N} |w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i|$$

Absolute loss is less common than squared loss, but can be very useful.  The main difference between the two is that training a model on absolute loss will produces estimates of the median of the label given the features.  Squared loss estimates the mean, and absolute loss estimates the median.  Whether you want to estimate the mean or median will depend on your use case.  Let's look at a few examples:
* If an error of -2 costs you \$2 and an error of +50 costs you \$50, then absolute loss models your costs better than squared loss.  
* If an error of -2 costs you \$2, while an error of +50 is simply unacceptably large, then it's important that your errors are generally small, and so squared loss is probably the right fit.  
* If it's important that your predictions are too high as often as they're too low, then you want to estimate the median with absolute loss.  
* If outliers in your training data are having too much influence on the model, try switching from squared to absolute loss.  Large errors get a large amount of attention from absolute loss, but with squared loss, large errors get squared and become huge errors attracting a huge amount of attention.  If the error is due to an outlier, it might not deserve a huge amount of attention.

Absolute loss is sometimes also known as $\text{L}_1$ loss or least absolute error.  Read more about [absolute loss](https://en.wikipedia.org/wiki/Least_absolute_deviations) on wikipedia.

### Quantile loss

    predictor_type='regressor',
    loss='quantile_loss',
    quantile=0.9,
    
$$ \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N q(y_i - w_o - \mathbf{x_i}^\intercal \mathbf{w})^\text{+} + (1-q)(w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i)^\text{+} $$

$$ \text{where the parameter } q \text{ is the quantile you want to predict}$$

Quantile loss lets us predict an upper or lower bound for the label, given the features. To make predictions that are larger than the true label 90% of the time, train quantile loss with the 0.9 quantile. An example would be predicting electricity demand where we want to build near peak demand since building to the average would result in brown-outs and upset customers.  Read more about [quantile loss](https://en.wikipedia.org/wiki/Quantile_regression) on wikipedia.

### Huber loss

    predictor_type='regressor',
    loss='huber_loss',
    huber_delta=0.5,

$$ \text{Let the error be } e_i = w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i \text{.  Then Huber loss solves:}$$

$$ \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N I(|e_i| < \delta) \frac{e_i^2}{2} + I(|e_i| >= \delta) |e_i|\delta - \frac{\delta^2}{2} $$

$$ \text{where } I(a) = 1 \text{ if } a \text{ is true, else } 0 $$

Huber loss is an interesting hybrid of $\text{L}_1$ and $\text{L}_2$ losses.  Huber loss counts small errors on a squared scale and large errors on an absolute scale.  In the plot above, we see that Huber loss looks like squared loss when the error is near 0 and absolute loss beyond that.  Huber loss is useful when we want to train with squared loss, but want to avoid squared loss's sensitivity to outliers.  Huber loss gives less importance to outliers by not squaring the larger errors.  Read more about [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) on wikipedia.

### Epsilon-insensitive loss

    predictor_type='regressor',
    loss='eps_insensitive_squared_loss',
    loss_insensitivity=0.25,
    

For epsilon-insensitive squared loss, we minimize
$$ \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N max(0, (w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i)^2 - \epsilon^2) $$

And for epsilon-insensitive absolute loss, we minimize

$$ \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N max(0, |w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i| - \epsilon) $$

Epsilon-insensitive loss is useful when errors don't matter to you as long as they're below some threshold.  Set the threshold that makes sense for your use case as epsilon.  Epsilon-insensitive loss will allow the model to pay no cost for making errors smaller than epsilon.

### Logistic regression

    predictor_type='binary_classifier',
    loss='logistic',
    binary_classifier_model_selection_criteria='recall_at_target_precision',
    target_precision=0.9,

Each of the losses we've discussed is for regression problems, where the labels are floating point numbers.  The last two losses we'll cover, logistic regression and support vector machines, are for binary classification problems where the labels are one of two classes.  Linear learner expects the class labels to be 0 or 1.  This may require some preprocessing, for example if your labels are coded as -1 and +1, or as blue and yellow.  Logistic regression produces a predicted probability for each data point:

$$ p_i = \sigma(w_0 + \mathbf{x_i}^\intercal \mathbf{w}) $$

The loss function minimized in training a logistic regression model is the log likelihood of a binomial distribution.  It assigns the highest cost to predictions that are confident and wrong, for example a prediction of 0.99 when the true label was 0, or a prediction of 0.002 when the true label was positive.  The loss function is:

$$ \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N y_i  \text{log}(p) - (1 - y_i) \text{log}(1 - p) $$  

$$ \text{where } \sigma(x) = \frac{\text{exp}(x)}{1 + \text{exp}(x)}  $$

Read more about [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) on wikipedia.

### Hinge loss (support vector machine)

    predictor_type='binary_classifier',
    loss='hinge_loss',
    margin=1.0,
    binary_classifier_model_selection_criteria='recall_at_target_precision',
    target_precision=0.9,
    
Another popular option for binary classification problems is the hinge loss, also known as a Support Vector Machine (SVM) or Support Vector Classifier (SVC) with a linear kernel.  It places a high cost on any points that are misclassified or nearly misclassified.  To tune the meaning of "nearly", adjust the margin parameter:

It's difficult to say in advance whether logistic regression or SVM will be the right model for a binary classification problem, though logistic regression is generally a more popular choice then SVM.  If it's important to provide probabilities of the predicted class labels, then logistic regression will be the right choice.  If all that matters is better accuracy, precision, or recall, then either model may be appropriate.  One advantage of logistic regression is that it produces the probability of an example having a positive label.  That can be useful, for example in an ad serving system where the predicted click probability is used as an input to a bidding mechanism.  Hinge loss does not produce class probabilities.

Whichever model you choose, you're likely to benefit from linear learner's options for tuning the threshold that separates positive from negative predictions

$$\text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^{N} y_i(\frac{m+1}{2} - w_0 - \mathbf{x_i}^\text{T}\mathbf{w})^\text{+} + (1-y_i)\frac{m-1}{2} + w_o + \mathbf{x_i}^\text{T}\mathbf{w})^\text{+}$$

$$\text{where  } a^\text{+} = \text{max}(0, a)$$


Note that the hinge loss we use is a reparameterization of the usual hinge loss: typically hinge loss expects the binary label to be in {-1, 1}, whereas ours expects the binary labels to be in {0, 1}.  This reparameterization allows LinearLearner to accept the same data format for binary classification regardless of the training loss.  Read more about [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) on wikipedia.


## Class weights
In some binary classification problems, we may find that our training data is highly unbalanced.  For example, in credit card fraud detection, we're likely to have many more examples of non-fraudulent transactions than fraudulent.  In these cases, balancing the class weights may improve model performance.
 
Suppose we have 98% negative and 2% positive examples.  To balance the total weight of each class, we can set the positive class weight to be 49.  Now the average weight from the positive class is 0.98 $\cdot$ 1 = 0.98, and the average weight from the negative class is 0.02 $\cdot$ 49 = 0.98.  The negative class weight multiplier is always 1.
 
To incorporate the positive class weight in training, we multiply the loss by the positive weight whenever we see a positive class label.  For logistic regression, the weighted loss is:

Weighted logistic regression:

$$ \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N p y_i  \text{log}(\sigma(w_0 + \mathbf{x_i}^\intercal \mathbf{w})) - (1 - y_i) \text{log}(1 - \sigma(w_0 + \mathbf{x_i}^\intercal \mathbf{w})) $$  

$$ \text{where } p \text{ is the weight for the positive class.} $$
 
The only difference between the weighted and unweighted logistic regression loss functions is the presense of the class weight, $p$ on the left-hand term in the loss.  Class weights in the hinge loss (SVM) classifier are applied in the same way.

To apply class weights when training a model with linear learner, supply the weight for the positive class as a training parameter:

    positive_example_weight_mult=200,

Or to ask linear learner to calculate the positive class weight for you:

    positive_example_weight_mult='balanced',