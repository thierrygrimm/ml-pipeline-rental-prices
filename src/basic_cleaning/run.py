#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging

import pandas as pd

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Downloading artifact {args.input_artifact}")
    local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    # Drop outliers
    logger.info(f"Dropping prices below {args.min_price} or above {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price) & df['longitude'].between(-74.25, -73.50) & df[
        'latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime format")
    df['last_review'] = pd.to_datetime(df['last_review'])
    df.to_csv("clean_sample.csv", index=False)

    # Log cleaned data as artifact clean_sample.csv
    logger.info("Logging artifact of cleaned file")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input file to be processed",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact after cleaning",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="cleaned_file",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Dataset after cleaning",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The lower boundary to the price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The upper boundary to the price",
        required=True
    )

    args = parser.parse_args()

    go(args)
