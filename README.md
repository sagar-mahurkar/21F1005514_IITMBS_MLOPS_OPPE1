# SEPT_2025_OPPE_1

## Assignment Objectives

1. Set up Data versioning using DVC for the 2 versions of provided data 

    - Configure Google Cloud Storage as Remote Storage Backend for DVC 

2. Integrate and Utilize Feast Feature store for storing/serving features.

3. Execute training and evaluation scripts producing valid predictions for 2 iterations of incremental data scenarios

4. Integrate Hyperparameter tuning and experiment tracking using MLflow.

    - Upload best model to MLFlow model registry 

5. Configure CI on the main branch to predict and generate metrics on test data 

    - Fetch best model from MLFlow

    - Test data needs to be fetched from DVC

    - with at least 1 sanity test per feature and report generation using CML on GitHub.
   

## Approach

1. Create script for generating preprocessed data

2. Prepare model training algorithm

3. Configure the data with dvc and train the model