# stickleback

A Python package for finding *point behaviors* in longitudinal sensor data.

## Installation of Pre-reqs

- Install [GitHub Desktop](https://desktop.github.com/)
- Clone [the repo](https://github.com/FlukeAndFeather/stickleback)
- Install [Python 3](https://www.python.org/downloads/)
- Install Pip: `python3 -m pip install --upgrade pip setuptools wheel`
- Update your PATH as needed (you'd have been warned during pip install)
- Install [Node](https://nodejs.org/en/download/)
- Install [NPM](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
- Install Yarn: `npm install -g yarn`
- Install Node modules: `yarn install`
- Install [VS Code](https://code.visualstudio.com/download)
- Install [Python VS Code extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- Run `pip install bumpversion`
- Run `pip install stickleback`
- Run `pip install black`
- Run `pip install pytest`

Developer inner-loop: 

1. Open `jupyter/README.ipynb`
1. Run cells
1. Modify `stickleback/stickleback.py` and other files
1. Change the version number in `stickleback/__init__.py`
1. `python3 setup.py sdist bdist_wheel`
1. `pip install --upgrade --force-reinstall stickleback`

Note version "0.0.0" below rather than desired upgrade: "0.0.2":
```
$ pip install --upgrade stickleback
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: stickleback in /Users/brodyberg/Library/Python/3.8/lib/python/site-packages/stickleback-0.0.0-py3.8.egg (0.0.0)
```
And force upgrade looks at egg, not wheel: 

```
$ pip install --upgrade --force-reinstall stickleback
Defaulting to user installation because normal site-packages is not writeable
ERROR: Could not find a version that satisfies the requirement stickleback (from versions: none)
ERROR: No matching distribution found for stickleback
```

---

## Begin Notebook

```python
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stickleback.stickleback import Stickleback
plt.rcParams['figure.figsize'] = [12, 8]
```

## Usage

Read the example data (longitudinal sensor data and events) and visualizing them.


```python
# Read example data
breath_sb = Stickleback(
    sensors=pd.read_pickle("../data/bw180828-49_prh10.pkl"), 
    events=pd.DatetimeIndex(pd.read_pickle("../data/bw180828-49_breaths.pkl")),
    win_size=50, min_period=10
)

ex_data_plot = breath_sb.plot_sensors_events(interactive=False)
```


    
![png](README_files/README_4_0.png)
    


Create the training dataset using all known events and an equal sample size of randomly selected non-events. This dataset must be in `sktime`-compatible nested `DataFrame` format.


```python
breath_sb.sample_nonevents()
print("+: {}\n-: {}".format(breath_sb.event_idx, breath_sb.nonevent_idx))
breath_sb.extract_training_data()
```

    +: [ 6595  6811  6955  7110  7245  7418  7564  7737  9579  9911 10148 17936
     18057 18197 18318 18454 18567 18705 18836 18994 19144 19333 19523 19824
     28917 29049 29188 29337 29473 29627 29747 29935 30645 30860 31083 31351
     31619]
    -: [ 4600  4700  4750  8350  8800  9100  9500  9700 10750 11400 12500 12550
     13350 13900 14050 14350 15000 15850 16550 16950 17500 17850 19250 19650
     20600 21450 21950 22550 22850 23850 27350 28450 30250 31150 31800 33450
     34450]


Using the training data: fit the model, make predictions, and assess in-sample accuracy.


```python
breath_sb.fit()
breath_sb.predict_self(nth=5)
breath_sb.assess(tol=40)
tp, fp, fn = (np.sum(breath_sb.outcomes == o) for o in ["TP", "FP", "FN"])
print("TP: {}, FP: {}, FN: {}".format(tp, fp, fn))
first_predictions = breath_sb.plot_predictions(interactive=False)
```

    TP: 37, FP: 143, FN: 0



    
![png](README_files/README_8_1.png)
    


The randomly sampled non-events are unlikely to contain much useful information for differentiating windows with events from windows that *almost* look like events. That's why the first round of predictions resulted in many true positives and few false negatives, but many false positives as well. The false positives are *almost* events that we use to refine the model.


```python
breath_sb.refit()
breath_sb.predict_self(nth=5)
breath_sb.assess(tol=40)
tp, fp, fn = (np.sum(breath_sb.outcomes == o) for o in ["TP", "FP", "FN"])
print("TP: {}, FP: {}, FN: {}".format(tp, fp, fn))
second_predictions = breath_sb.plot_predictions(interactive=False)
```

    TP: 31, FP: 1, FN: 6



    
![png](README_files/README_10_1.png)
    



```python

```
