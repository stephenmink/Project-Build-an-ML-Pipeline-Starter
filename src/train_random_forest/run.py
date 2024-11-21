#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column.
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Load Random Forest configuration
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
        run.config.update(rf_config)

    # Fix the random seed for reproducibility
    rf_config['random_state'] = args.random_seed

    # Load training and validation artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    df = pd.read_csv(trainval_local_path)

    
    # Data Preprocessing: Handle object columns and NaNs
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    df = df.dropna(subset=["price", "name", "last_review", "longitude", "latitude"])

    # Drop rows with any NaN values
    logger.info(f"Shape before dropping NaNs: {df.shape}")
    df = df.dropna()
    logger.info(f"Shape after dropping NaNs: {df.shape}")

    # Split target and features
    y = df.pop("price")
    X = df.copy()

    logger.info(f"Dataset after dropping rows with NaNs in critical columns:\n{df.isna().sum()}")
    assert not df.isna().any().any(), "There are still NaN values in the dataset after dropping critical columns!"


    # Price Validation
    min_price, max_price = y.min(), y.max()
    if min_price < args.min_price or max_price > args.max_price:
        raise ValueError(f"Price out of bounds: {min_price} < {args.min_price} or {max_price} > {args.max_price}")
    logger.info(f"Minimum price: {min_price}, Maximum price: {max_price}")

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    # Build and validate pipeline
    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    logger.info("Validating preprocessing pipeline")

    # Validate columns required by the pipeline
    preprocessor = sk_pipe.named_steps["preprocessor"]
    expected_columns = []
    for _, _, columns in preprocessor.transformers:
        if isinstance(columns, list):
            expected_columns.extend(columns)
        elif isinstance(columns, str):
            expected_columns.append(columns)

    missing_columns = set(expected_columns) - set(X.columns)
    if missing_columns:
        logger.error(f"Missing columns in training data: {missing_columns}")
        raise ValueError(f"Columns missing in training data: {missing_columns}")

    # Check for NaNs after preprocessing
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    if np.isnan(X_train_transformed).any() or np.isnan(X_val_transformed).any():
        raise ValueError("NaN values found in preprocessed data.")

    # Fit the pipeline
    logger.info("Fitting the model")
    sk_pipe.fit(X_train, y_train)

    # Evaluate the model
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"R-squared: {r_squared}")
    logger.info(f"Mean Absolute Error: {mae}")

    # Save the model
    logger.info("Exporting model")
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")
    signature = mlflow.models.infer_signature(X_val, y_pred)
    mlflow.sklearn.save_model(
        sk_pipe,
        path="random_forest_dir",
        signature=signature,
        input_example=X_train.iloc[:5],
    )

    # Upload the model to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Trained Random Forest artifact",
        metadata=rf_config,
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)
    run.summary["r2"] = r_squared
    run.summary["mae"] = mae
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})


def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names) - 1]
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)

    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, align="center")
    sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OrdinalEncoder()
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude",
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    date_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="2010-01-01"),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False),
    )
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        FunctionTransformer(np.reshape, kw_args={"newshape": -1}),
        TfidfVectorizer(max_features=max_tfidf_features, stop_words="english"),
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"]),
        ]
    )
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]
    random_forest = RandomForestRegressor(**rf_config)

    sk_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("random_forest", random_forest)])
    return sk_pipe, processed_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest")
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--stratify_by", type=str, default="none")
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, default=10)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)
    args = parser.parse_args()

    go(args)
