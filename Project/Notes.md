# Notes

## Style notes (from pylint warnings, to respect pep8)

TODO

- It's okay and even better to have more expressive class names. Explicit is better than implicit. (see [PEP 20 -- The Zen of Python](https://www.python.org/dev/peps/pep-0020/))

DONE

- Don't leave trailing whitespace (I fixed it for now with quick use of an extension)
- Always leave a blank newline a end of files
- Either use snake_case or PascalCase naming style, don't use both --> snake_case
- There is a certain order to imports (with blank line between each group):
  - standard library
  - related third party imports
  - local application/library specific imports
  - See : [Importing modules in Python - recommended position](https://stackoverflow.com/questions/20411014/importing-modules-in-python-recommended-position)
- Class names need to start with a capital letter.

## Clarity notes

TODO

- (in models) model : model_name
- (in models) parameters : parameters_range
- classifier :
  - "training the datasets" does not mean the right thing. "Training on the datasets" would be more accurate.
  - the args in the init are wrong, there are more than the actual number of parameters. The extra info should be attributes in the general class docstring.
  - change "test" for "valid" when applicable
- Docstrings for everything

DONE

- preprocessing : "trainTestSplit" not an ideal function names. Could be called "TrainValidSplit".

## Usage notes

TODO

- One should be able to fix different hyperparameters than the ones found by the optimisation. We can't let everything be done in the init.
  - This means we need to have an attribute for each hyperparameters, and the default values would either be what we found with optimisation, or "None" (and then they need to be fixed by the user or the automatic optimisation)

- Getter/setter for individual hyperparameters.

- Training possible without grid search.

- There is no cross-validation done right now! Oh no :O ! Big oopsie! "trainTestSplit" only returns one (1) split from the k-fold split.

- preprocessing/split : Remove "random_state=42"

- Once hyperparam optimization done, get generalisation score on test set!!!

DONE

## Report/analysis notes

- We need to show as much as we can what PCA does in the report, meaning we need more data on its effects. We need to show we pretreated the data in a smart way.
  - For example, what's a more optimal amount of principal components? When does it increase performance and when does it decrease it, and does it affect each classifier the same way? And Why? --> OUI
  - What happens if we apply the PCA on only some sets of features, and not all features together (after all, each data point is 3 sets of 64 values, and each set has a different meaning) (someone on Kaggle did some PCA analysis, we can use that as inspiration) --> EXTRA IF WE HAVE MUCH TIME LEFT
- We need to show some results from the grid search optimisation. Why did we choose the grid we did with each classifier? --> Perceptron
- What happens if we do GridSearch with a different scoring method???

## Information I need

- What kind of results did we have with current setup? How much time did it take to run? Is k=10 cross-validation realistic or should we try k=5?
  - In general more than 90%. Perceptron around 50%.
  - All classifiers under 30s.

## How hyperparams were obtained

### Logistic regression

(15min)

First search large (100 configs k=5)
"C" : np.linspace(0.1, 10, num=10)
"l1_ratio" : np.linspace(0, 1, num=10)

The model : Logistic Regression
The best parameters : {'penalty': 'elasticnet', 'l1_ratio': 0.4444444444444444, 'C': 7.800000000000001}
Training accuracy: 0.9966329966329966
Validation accuracy: 0.9595959595959596
Training f1-score: 0.9966014635519279
Validation f1-score: 0.9478114478114479

Second search (25 configs, k=5)
"C" : np.linspace(1, 15, num=5),
"l1_ratio" : np.linspace(0.4, 0.6, num=5)
The model : Logistic Regression
The best parameters : {'l1_ratio': 0.4, 'C': 8.0, 'penalty': 'elasticnet'}
Training accuracy: 0.9966329966329966
Validation accuracy: 0.9595959595959596
Training f1-score: 0.9966014635519279
Validation f1-score: 0.9478114478114479
