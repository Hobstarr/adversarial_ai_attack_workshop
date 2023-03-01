# How to get started
### On Windows:
1) Clone the repository onto your local machine, either by using GitHub Desktop app or by typing "git clone https://github.com/Hobstarr/dmu_workshop.git" in the command line
2) Open the repository folder using e.g. Visual Studio Code (recommended)
3) Open the terminal within VS Code, and start the virtual environment with the command "python3 -m venv workshop" (replace python3 with different version if needed)
4) Make sure you're using the recommended python interpreter (in VS Code: Ctrl+Shift+P, then find "Python: Select Interpreter", and select the recommended one)
5) Activate the virtual environment by typing "workshop\Scripts\activate" (in VS Code: if you get "Permission denied" error, click on the dropdown next to the "+" sign located in the top-right of the terminal window, and make sure "Command Prompt" is selected)
6) At this point you're ready to run the code step-by-step, by selecting a part of the code in main.py and running it with shift+enter

### On Mac/Linux:
1) Clone the repository onto your local machine. [recommended method:] 
- Open VSCode and use Powershell Command (command+shift+p / ctrl+shift+p) 'Clone from Github', copying the following location when prompted: 'https://github.com/Hobstarr/dmu_workshop.git'.
- Choose a repository location on your computer '/PythonProjects/dmu_workshop' for instance.
- Follow the prompts accepting that you want to open this repository

2) Create virtual environment. [recommended method:] 
- Open VSCode and use Powershell Command (command+shift+p / ctrl+shift+p) 'Python: Create Virtual Environment'
- Choose Venv
- Choose Python 3.11.2
- Click on requirements.txt box and click ok (if this doesn't work, retry but don't click the requirements.txt box, refresh terminal (terminal, new terminal), try pip install -r requirements.txt in the venv, or pip install library (for each library at the start of main.py))

# Adversarial Workshop - FAQ
## ML attack fundamentals
### What is a model? 
function approximation method of (joint probability X, y).
For some data/input (X), we are trying to learn to best predict it's label or output (y).
We are aiming to optimise this over a whole set of inputs and outputs with respect to some statistc.

### What is Tensorflow? - 'an optimised machine 
learning platform for model creation and dataset management
with high computational speed and efficiency' 
-alternatives include: PyTorch / Jax

### Assuming we have a model, will it continue to work as expected?
As long as the underlying relationship between X and y doesn't change, and the distribution of X doesn't affect the models ability to predict the underlying relationship

### How can a normal model be tricked?
There are many ways to affect the availability of a model, but we want to affect the integrity by reducing it's efficacy, or confidence in it's predictions.

### What is an adversarial example?
An adversarial example is an example that has been editted, so that X + delta is classified in a different way than X, with some small delta. (There are other ways where this isn't exactly true, but this is generally true).

### How does gradient tape work?
GradientTape is an auto-differentiation or auto-gradient method that follows a function and returns it's gradient.
We can use simple functions that we know, or follow a complex function as long as it runs through tensorflow.
This means that ML models can be followed, for instance we can find the gradient of the input with respect to the loss for a certain label.

Some notes: 
model.fit uses gradient in training_step to update model.trainable_variables.
model(x) can be followed, model.predict(x)

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

 cannot
            trade_off between clean_acc and adv_acc, but more resilient vs attack.
            [in some cases acts as regularisation method]


