# European Soccer

Data Analysis and Machine Learning with Kaggle European Soccer Database

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Dataset

[Kaggle European Soccer](https://www.kaggle.com/hugomathien/soccer)

### Built With

* Python 3.6.10

* numpy >= 1.16.2

* matplotlib >= 3.1.1

* PyTorch >= 1.7.0

* scikit-learn >= 0.23.1

* pandas >= 1.0.5

## Code Organization
```
.
├── src                     # Python scripts
│   ├── value_iteration.py  # VI algorithm
│   ├── policy_iteration.py # PI algorithm
│   ├── q_learning.py       # Q learning algorithm
│   └── utils.py            # Utility sets
├── european_soccer.ipynb   # Visualization Results
└── README.md
```

## Tests

There are 3 methods you can try, namely policy iteration, value iteration, and Q learning, with corresponding file name.

ex. if you want to try policy iteration, just do
```
python policy_iteration.py
```

## Results

Please refer to the [notebook](https://github.com/arthur960304/european_soccer/blob/main/european_soccer.ipynb) to see the result.
