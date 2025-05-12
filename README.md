# SUDOKU


## File structure
```
├── README.md          <- The top-level README for developers using this project
├── config.yml         <- Configuration file
├── requirements.txt   <- The requirements file for reproducing the environment
├── .gitignore         <- Git ignore file
├── data/
│   ├── raw/           <- The original, immutable data dump
│   ├── interim/       <- Intermediate data that has been transformed
│   └── processed/     <- The final, canonical data sets for modeling
│
├── docs/              <- Any descriptors of your data or models
│
├── models/            <- Trained models (so you don't have to rerun them)
│
├── notebooks/         <- Jupyter notebooks for temporary EDA, exploration, etc.
│
├── reports/           <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/       <- Generated graphics and figures to be used in reporting
│
├── results/           <- Saved model outputs and/or metrics
│
└── src/               <- Source code for use in this project
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data/          <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── preprocess/    <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── model/         <- Scripts to train models and apply models  
    │   ├── predict_model.py
    │   └── train_model.py
    │
    ├── evaluate/      <- Scripts to validate and apply the model to data  
    │
    ├── visualization/ <- Scripts to create exploratory and results oriented visualizations
    │   └── visualize.py
    │
    └── common/        <- Scripts shared among other modules
```


