# Food & You - Data Aggregator

A data processing pipeline for aggregating and aligning continuous glucose monitoring (CGM) data with food intake records from the Food & You study.

## Overview

This pipeline performs the following key operations:

1. Loads participant data, food records (MFR), and glucose measurements from the database
2. Filters for completed study participants  
3. Resamples CGM data at 15-minute intervals with interpolation
4. Aggregates food intake events by dish, then aligns them with CGM timestamps
5. Exports the final merged dataset for downstream analysis

## Key Features

- Handles timezone-aware datetime interpolation
- Splits CGM timeseries based on significant gaps (>2 hours)
- Aggregates multiple food items into single intake events  
- Aligns food intake timestamps with nearest CGM readings
- Exports both full dataset and development subset
- Rich logging with progress tracking

## Requirements

- Python 3.x
- pandas
- numpy 
- scipy
- tqdm
- python-dotenv
- rich (for logging)

## Setup

1. Create a `.env` file in the root directory with the required database credentials and API keys (see `.env.example`)
2. Install dependencies:
```bash

conda install conda-forge::r-base # install R (for the IGLU metrics)
pip install -r requirements.txt
```

# db_helpers

Ensure that your db_helpers has all the necessary files as in the structure below:
```
.
├── data
│   ├── processed
│   │   ├── v0.2 (same as the latest version)
│   │   ├── v0.3 (same as the latest version)
│   │   ├── v0.4 (same as the latest version)
│   │   └── v0.5 
│   │       ├── debug
│   │       │   ├── debug-dishes-data-v0.5.csv
│   │       │   ├── debug-fay-ppgr-processed-and-aggregated-v0.5.csv
│   │       │   ├── debug-food_aggregated.csv
│   │       │   ├── debug-food-embeddings-v0.5.csv
│   │       │   ├── debug-food-embeddings-v0.5.csv.gz
│   │       │   ├── debug-glucose_raw.csv
│   │       │   ├── debug-glucose_resampled.csv
│   │       │   ├── debug-microbiome-data-v0.5.csv
│   │       │   └── debug-users-demographics-data-v0.5.csv
│   │       ├── dishes-data-v0.5.csv
│   │       ├── fay-ppgr-processed-and-aggregated-v0.5.csv
│   │       ├── food_aggregated.csv
│   │       ├── food-embeddings-v0.5.csv.gz
│   │       ├── glucose_raw.csv
│   │       ├── glucose_resampled.csv
│   │       ├── microbiome-data-v0.5.csv
│   │       └── users-demographics-data-v0.5.csv
│   └── raw
│       ├── food_embeddings_v0.1-debug.pkl.gz
│       ├── food_embeddings_v0.1.pkl.gz
│       └── user_data
│           ├── CHANGELOG.txt
│           ├── microbiome_metadata.csv
│           └── user_microbiome.tsv.zip
├── DataAggregator-v0.4.py
├── db_helpers
│   ├── cache
│   │   ├── CHANGELOG.txt
│   │   ├── mfr_data_v0.csv
│   │   └── mfr_data_v1.csv
│   ├── database.py
│   ├── data_sources.py
│   ├── db
│   │   └── sql
│   │       ├── db_table_exists.sql
│   │       ├── fanal_glucose_filtered.sql
│   │       ├── fanal_participants_tracked.sql
│   │       ├── fanal_sleep_all.sql
│   │       └── fay_sleep_all.sql
│   ├── __init__.py
│   └── __pycache__
│       ├── database.cpython-311.pyc
│       ├── data_sources.cpython-311.pyc
│       └── __init__.cpython-311.pyc
├── generate_food_embeddings.py
├── notebooks
│   ├── data
│   │   └── processed
│   │       └── v0.4
│   │           └── debug
│   │               ├── debug-food_aggregated.csv
│   │               ├── debug-glucose_raw.csv
│   │               └── debug-glucose_resampled.csv
│   └── DataAggregator-v0.2.ipynb
├── README.md
├── requirements.txt
├── utils.py
└── versions
    ├── DataAggregator-v0.2.py
    └── DataAggregator-v0.3.py
```
You might need to talk to someone at DE lab to get access to the `mfr_data.csv` file.

## Usage

Run the data aggregation pipeline:

```bash
python DataAggregator-v0.2.py
```

## Output

The pipeline generates several CSV files in the configured output directory:

1. `glucose_resampled.csv` - Resampled CGM data
2. `food_aggregated.csv` - Aggregated food intake events  
3. `fay-ppgr-processed-and-aggregated-{VERSION}.csv` - Final merged dataset
4. `fay-ppgr-processed-and-aggregated-{VERSION}-dev.csv` - Development subset

## Food Embeddings

We use OpenAI's API to generate embeddings for the food items. The embeddings are saved in the `data/raw/food_embeddings_{VERSION}.csv` file.
The versions are as follows: 
- `v0.1` - All unique food_ids in `./db_helpers/cache/mfr_data_v1.csv` using OpenAI `text-embedding-3-large`

## Author

* Sharada Mohanty (sharada.mohanty@epfl.ch)

## License

This project is part of the Food & You research initiative. All rights reserved.

## Contributing

For contributing to this repository, please first discuss the change you wish to make via issue or email with the repository owners.