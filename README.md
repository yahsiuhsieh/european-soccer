# European Soccer

<p align="center">
  <img width="500" height="400" src="https://github.com/arthur960304/european_soccer/blob/main/dataset-cover.jpg">
</p>

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
├── src                         # Python scripts
│   ├── prediction              # Prediction scripts
│   │   ├── utils.py            # Utility set for prediction
│   │   ├── sklearn_test.py     # Script for testing sklearn model
│   │   ├── models.py           # Define PyTorch objects
│   │   └── soccernet.py        # Neural network training and prediction
│   ├── visualization           # Visualization scripts
│   │   ├── pca_kmeans.py       # Perform PCA and Kmeans to the data
│   │   ├── visual_utils.py     # Utility set for visualization
│   └── └── visual.py           # Data visualization script
├── european_soccer.ipynb       # Notebook showing all results
└── README.md
```

## Tests

There are two ways you can do to get the results.

1. Directly run the jupyter notebook (suggested)
2. Run the scripts (See section [About the scripts](#about-the-scripts) to get more info)

### About the scripts

There are four files you can run

* run [sklearn_test.py](https://github.com/arthur960304/european_soccer/blob/main/src/prediction/sklearn_test.py) to test the result of sklearn model
* run [soccernet.py](https://github.com/arthur960304/european_soccer/blob/main/src/prediction/soccernet.py) to train and test the neural network
* run [pca_kmeans.py](https://github.com/arthur960304/european_soccer/blob/main/src/visualization/pca_kmeans.py) to see how PCA and KMeans perform on this dataset
* run [visual.py](https://github.com/arthur960304/european_soccer/blob/main/src/visualization/visual.py) to see the basic data analysis

## Results

Please refer to the [notebook](https://github.com/arthur960304/european_soccer/blob/main/european_soccer.ipynb) to see the result.
