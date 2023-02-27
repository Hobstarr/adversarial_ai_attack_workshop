# Adversarial Workshop - DMU Cyberweek
What is Tensorflow? - 'an optimised machine 
learning platform for model creation and dataset management
with high computational speed and efficiency' 
-alternatives include: PyTorch / Jax

# ML attack fundamentals
### What is a model? 
function approximation method (joint probability X, y)

### Assuming we have a model, will it continue to work as expected?
As long as the underlying relationship between X and y doesn't change, and the distribution of X doesn't affect the models ability to predict the underlying relationship

### How can a normal model be tricked?
There are many ways to affect the availability of a model, but we want to affect the integrity by reducing it's efficacy, or confidence in it's predictions.

### What is an adversarial example?
An adversarial example is an example that has been editted, so that X + delta is classified in a different way than X, with some small delta. (There are other ways where this isn't exactly true, but this is generally true).

How does gradient tape work?
Simple functions - that we know
Random functions - that we don't know
ML models can be followed - input wrt loss etc
model.fit uses gradient in training_step to update model.trainable_variables.

Visualisation of gradient tape for example/model.
Visualisation of adversarial example.

# ML attack methodology
Can an attack made for one model
attack other models? What kind of efficiency

What kind of heuristics can we expect?
What is adversarial training?
What kind of performance can we receive?

# ML defenses - adversarial training
Create a few random models - attack them
Create a model - adversarial train it - attack it
Can we change the perturbation of a model?

Some notes: model(x) can be followed, model.predict(x) cannot
            trade_off between clean_acc and adv_acc, but more resilient vs attack.
            [in some cases acts as regularisation method]


