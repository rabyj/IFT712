# Notes

## Style notes (from pylint warnings, to respect pep8)

DONE
- More expressive class names. Explicit is better than implicit. (see [PEP 20 -- The Zen of Python](https://www.python.org/dev/peps/pep-0020/))
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

DONE

- Naive Bayes mention PCA in docstring somewhere.
- preprocessing : "trainTestSplit" not an ideal function names. Could be called "TrainValidSplit".
- (in models) model : model_name
- (in models) parameters : parameters_range
- change "test" for "valid" when applicable
- classifier :
  - "training the datasets" does not mean the right thing. "Training on the datasets" would be more accurate.
  - the args in the init are wrong, there are more than the actual number of parameters. The extra info should be attributes in the general class docstring.
- Docstrings for everything

## Usage notes

DONE

- Classification of new data points

- One should be able to fix different hyperparameters than the ones found by the optimisation. We can't let everything be done in the init.

- CHECK IF SIGMA NEEDS TO HAVE *2 OR NOT!!!

- Use proper test set

- Separate preprocessing in different functions

- There is no cross-validation done right now! Oh no :O ! Big oopsie! "trainTestSplit" only returns one (1) split from the k-fold split.

- preprocessing/split : Remove "random_state=42"

- Once hyperparam optimization done, get generalisation score on test set!!! NOT APPLICABLE

## Report/analysis notes

- MAKE REPO PUBLIC!!!!

- We need to show as much as we can what PCA does in the report, meaning we need more data on its effects. We need to show we pretreated the data in a smart way.
  - For example, what's a more optimal amount of principal components? When does it increase performance and when does it decrease it, and does it affect each classifier the same way? And Why? --> OUI
  - What happens if we apply the PCA on only some sets of features, and not all features together (after all, each data point is 3 sets of 64 values, and each set has a different meaning) (someone on Kaggle did some PCA analysis, we can use that as inspiration) --> EXTRA IF WE HAVE MUCH TIME LEFT
- We need to show some results from the grid search optimisation. Why did we choose the grid we did with each classifier? --> Perceptron
- What happens if we do GridSearch with a different scoring method???

- don't forget to mention what we could have done more
  - see if applying pca by set of features affect results
  - properly integrating PCA to CV (particularly w Perceptron)
  - check for outliers
  - [Is it actually fine to perform unsupervised feature selection before cross-validation?](https://stats.stackexchange.com/questions/239898/is-it-actually-fine-to-perform-unsupervised-feature-selection-before-cross-valid)
  - [PCA and the train/test split](https://stats.stackexchange.com/questions/55718/pca-and-the-train-test-split)
  - using different scalers
  - technically we should apply data transformations from train to valid, not use valid for computing transformations

- Gaussian Naive Bayes : Show effect of pca. Compare effect of PCA with Perceptron!!!
