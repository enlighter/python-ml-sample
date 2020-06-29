# python-ml-sample
Sample Python Machine Learning Project

What it contains:
- ETL
  - Loads data from many source files and collates a single table containing the required information
  - Creates many derived features from existing columns for use in Machine Learning
  - Creates the feature set to be used by Machine Learning models
- Multi-stage modelling
  - First Classification is performed using Decision Tree Ensemble techiniques like Random Forest, XGBoost, etc.
  - On the Classification output performs Timeseries Regression using Decision Tree Ensemble techniques like Random Forest, XGBoost, etc., by using extended window modelling per new day
  - Ensembles output from several models trying to capture different aspects of the underlying data patterns
- Collates historical data and corresponding predictions and calculates performance of the modelling excercise
- Increases efficiency of time and resource utilisation by storing ETL outputs and/or trained models in Pickle dumps whenever the same output can be reused in a subsequent run
