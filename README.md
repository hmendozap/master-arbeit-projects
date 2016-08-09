# Auto-Net component for autosklearn and support files of HDMP's Master Thesis.

<!--Small repository of collection from various files used to write my thesis-->

Auto-Net is a tool for automatically configure neural networks using
[auto-sklearn](https://automl.github.io/auto-sklearn/stable/), a system for
automated machine learning. The result of this work was published as a workshop
paper: [Towards automatically-tuned neural
networks](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxhdXRvbWwyMDE2fGd4OjMzYjQ4OWNhNTFhNzlhNGE).
In this repository you could find the component that goes into auto-sklearn and
the plotting scripts used for hyperparameter analysis (requires fANOVA
dependency).

## Installation Requirements:
+ auto-sklearn
+ Theano (0.9.0rc1)
+ Lasagne (development branch)

Auto-sklearn uses scikit-learn's component methods, as Auto-Net does. These
components are located in inside autosk_dev_test/component/. Inside the
component's directory the implementation of the code in Theano can be found and
the unit tests.

To install inside auto-sklearn (recommended):

- Install auto-sklearn with editing capabilities (e.g. `pip -e `)
- Clone this repository
- Copy the file `autosk_dev_test/component/DeepNetIterative.py` to `path_to_autosklearn/auto-sklearn/autosklearn/pipeline/components/classification`
- Copy the file `autosk_dev_test/component/RegDeepNet.py` to `path_to_autosklearn/auto-sklearn/autosklearn/pipeline/components/regression`
- Copy the file `autosk_dev_test/component/implementation/FeedForwardNet.py` to `path_to_autosklearn/auto-sklearn/autosklearn/pipeline/implementations`
- Fix imports (actually just one line)

To only use auto-net inside autosklearn (Taken from [auto-sklearn
example](https://automl.github.io/auto-sklearn/stable/index.html#example)):

```python
automl = autosklearn.classification.AutoSklearnClassifier(include_estimators=['DeepNetIterative'])
```

To use auto-net as a third party component (Experimental):

- Install auto-sklearn
- Clone this repository
- *Optional*: Add the path of component to your PYTHONPATH

Add the component before starting autosklearn:

```python

from autosklearn.pipeline.components.classification import add_classifier
from component import DeepNetIterative

add_classifier(DeepNetIterative.DeepNetIterative)

[...]

automl = autosklearn.classification.AutoSklearnClassifier(include_estimators=['DeepNetIterative'])
```



<!--+ **plotting_param_distros**: Where I plotted the distributions of different parameters based on-->
<!--different solver types or preprocessing methods-->
