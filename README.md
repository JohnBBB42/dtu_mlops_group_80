# Renewable energy price prediction

A mlops project pipeline for predicting electricity prices of renewable energy

# Introduction
This repository contains the Machine Learning Operations (MLOps) pipeline for our exam project in the course Machine Learning Operations (02476).
The goal of our project is to analyze and predict renewable energy trends and energy price changes using machine learning. Our primary dataset is `intermittent-renewables-production-france.csv` by Henri Upton. This dataset, located under the `data/raw` directory, contains data about energy production across different times of the year and day, primarily in France, over the last four years. While this dataset will initially be used for training and testing, we may explore additional datasets as the project evolves and the pipeline becomes more robust.

# Framework
We plan to use PyTorch as the primary framework for building and training our machine learning models. Additionally, we may incorporate tools such as PyTorch Transformers (https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) for more advanced modeling. To manage the MLOps pipeline, we aim to use tools such as DVC (Data Version Control) and several other tools introduced in the course.

# Models
Initially, we will experiment with simpler models like feedforward neural networks (FNNs) to establish a baseline. Since we are dealing with time-series data, we plan to possibly extend this to more advanced models, such as:
Recurrent Neural Networks (RNNs): Designed to handle sequential data like time related data like in this project.
Transformers: A state-of-the-art architecture for sequential data, using PyTorch's Transformer library for implementation.
The goal is to predict energy price changes based on renewable energy production data. If successful, this approach could be applied to other countries beyond France.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

