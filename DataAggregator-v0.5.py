#!/usr/bin/env python
# coding: utf-8

"""
Single-run script for:

1. Loading participants, MFR (food) data, and glucose data from a database
   using the faydesc db_helpers modules.
2. Filtering to completed participants only.
3. Resampling CGM data at 15-minute intervals with interpolation.
4. Aggregating MFR data by dish, then aligning each food intake event to the 
   nearest CGM timestamp within a chosen time window.
5. Exporting the final merged dataset to CSV.

Author: [Your Name]
Date: [Date]
"""

import os
import re
import sys
import tqdm
from tqdm.rich import tqdm

import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from dotenv import load_dotenv
from scipy.stats import ttest_rel as paired_ttest

# Faydesc DB helpers 
from db_helpers import database as db
from db_helpers import data_sources as ds

import warnings
from tqdm import TqdmExperimentalWarning

import logging
from rich.logging import RichHandler

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
pd.set_option('future.no_silent_downcasting', True)
# ---------------------------------------------------------------------------
# ENVIRONMENT AND GLOBAL CONFIG
# ---------------------------------------------------------------------------

load_dotenv()

VERSION = "v0.5"

DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"
NUM_PARTICIPANTS_IN_DEBUG_MODE = 5

OUTPUT_DIRECTORY = f"data/processed/{VERSION}"

RESAMPLE_INTERVAL = "15min" # resample CGM data at 15-minute intervals

GLUCOSE_TIMESERIES_GAP_SPLIT_THRESHOLD = pd.Timedelta(hours=0.5) # break up all timeseries with gaps > 1 hours (else impute)
FOOD_ALIGNMENT_TIME_WINDOW = pd.Timedelta(hours=0.5)  # Max offset for linking MFR to CGM - at most +/- 30 minutes

INCLUDE_SLEEP_DATA = False
INCLUDE_ACTIVITY_DATA = False

MICROBIOME_DATA_PATH = "data/raw/user_data/user_microbiome.tsv.zip" # Rohan has access to this file
MICROBIOME_MIN_RELATIVE_ABUNDANCE = 0.0005
MICROBIOME_MIN_PRESENCE_FRACTION = 0.05


FOOD_EMBEDDINGS_PATH = "data/raw/food_embeddings_v0.1.pkl.gz"

if DEBUG_MODE:
    OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "debug")
    FILENAME_PREFIX = "debug-"
else:
    FILENAME_PREFIX = ""

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("rich")
log.warning(f"Debug mode: {DEBUG_MODE}")

# ---------------------------------------------------------------------------
# 1. LOAD DATA DIRECTLY FROM DATABASE
# ---------------------------------------------------------------------------

# A) Get all participants who completed the study
log.info("Querying completed participants...")
completed_participants = db.AnalyticsDb().select("""
    SELECT id, cohort
    FROM faydesc_participants
    WHERE phase = '4_study_end'
""")
participant_ids = set(completed_participants["id"])

if DEBUG_MODE:
    # In case of debug mode, only process the first 5 participants + 2 hand picked participants which have activity data
    participant_ids = sorted(list(participant_ids)[:NUM_PARTICIPANTS_IN_DEBUG_MODE]) + [2680, 5] 
    log.info(f"Debug mode: processing participants {participant_ids}")

# B) Get MFR data for these participants
log.info("Querying MFR (food) data for completed participants...")
all_mfr_data = ds.Data.get_dishes()  # Replace with your actual function or query
mfr_mask = all_mfr_data["fay_user_id"].isin(participant_ids)
filtered_mfr_data = all_mfr_data[mfr_mask].copy()

# C) Get glucose data
#    ds.Data.get_glucose_query() might return a Query object with .text, .parameters, and .identifiers
raw_glucose_path = os.path.join(OUTPUT_DIRECTORY, f"{FILENAME_PREFIX}glucose_raw.csv")
if not os.path.exists(raw_glucose_path):
    log.info("Querying glucose data...")
    glucose_query = ds.Data.get_glucose_query()
    raw_glucose_df = db.AnalyticsDb().select(
        q=f"SELECT g.* FROM ({glucose_query.text}) g",
        p=glucose_query.parameters,
        i=glucose_query.identifiers
    )
else:
    log.warning(f"Loading raw glucose data from: {raw_glucose_path}")
    raw_glucose_df = pd.read_csv(raw_glucose_path)

if DEBUG_MODE:
    # Only use a subset of the participants in debug mode
    raw_glucose_df = raw_glucose_df[raw_glucose_df["user_id"].isin(participant_ids)]

# Store raw glucose data before processing
raw_glucose_df["read_at"] = pd.to_datetime(raw_glucose_df["read_at"]) # ensure read_at is datetime
raw_glucose_df["loc_read_at"] = pd.to_datetime(raw_glucose_df["loc_read_at"]) # ensure loc_read_at is datetime

raw_glucose_df.to_csv(raw_glucose_path, index=False)
log.info(f"Raw glucose data saved to: {raw_glucose_path}")

# ---------------------------------------------------------------------------
# 2. RESAMPLE GLUCOSE DATA (15-MIN INTERVAL)
# ---------------------------------------------------------------------------

def interpolate_datetime_column(df: pd.DataFrame,
                                datetime_col: str,
                                index_name: str) -> pd.Series:
    """
    Interpolates a datetime column using the current DateTimeIndex for time-based 
    interpolation. This function:
      1) Converts a tz-aware series to UTC, removing the timezone.
      2) Converts non-null datetimes to int64 representation, with NaTs as NaN.
      3) Interpolates (method='time') over that numeric representation.
      4) Converts back from numeric into datetime64.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the datetime column to be interpolated.
    datetime_col : str
        Name of the datetime column in df to interpolate.
    index_name : str
        Name of the DataFrame's DateTimeIndex (e.g., "read_at").

    Returns
    -------
    pd.Series
        The interpolated datetime Series.
    """
    from pandas.core.dtypes.dtypes import DatetimeTZDtype

    # Return original column if it does not exist:
    if datetime_col not in df.columns:
        return df.get(datetime_col, pd.Series(index=df.index, dtype='datetime64[ns]'))

    datetime_series = df[datetime_col]

    # If timezone-aware, convert to UTC then remove timezone
    if isinstance(datetime_series.dtype, DatetimeTZDtype):
        datetime_series = datetime_series.dt.tz_convert("UTC").dt.tz_localize(None)

    # Prepare a float Series for interpolation
    numeric_series = pd.Series(index=df.index, dtype="float64")
    not_na_mask = datetime_series.notna()
    
    # Convert datetimes to int64 (nanoseconds), keep NaT as NaN
    numeric_series[not_na_mask] = datetime_series[not_na_mask].astype("int64").astype(float)

    # Attach temporary numeric column to df for interpolation
    df["__temp_numeric"] = numeric_series
    df["__temp_numeric"] = df["__temp_numeric"].interpolate(method="time")

    # Convert back from float -> int -> datetime64
    df["__temp_numeric"] = df["__temp_numeric"].round().astype("Int64")  # allows <NA>
    df["__temp_datetime"] = pd.to_datetime(df["__temp_numeric"], errors="coerce")

    # Clean up
    df.drop(columns=["__temp_numeric"], inplace=True)
    final_interpolated = df["__temp_datetime"]
    df.drop(columns=["__temp_datetime"], inplace=True)

    return final_interpolated


total_duplicate_count = 0

def resample_cgm_for_user(user_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits CGM data (for one user) into blocks based on time gaps (>2h),
    resamples at 15-min intervals, and interpolates glucose values and 
    local read_at times.
    """
    global total_duplicate_count

    # Ensure sorted by device timestamp
    user_df = user_df.sort_values("read_at")

    # Identify blocks where gaps > 2h
    user_df["gap_indicator"] = (
        abs(user_df["read_at"].diff()) > GLUCOSE_TIMESERIES_GAP_SPLIT_THRESHOLD
    ).cumsum()

    user_id = user_df["user_id"].iloc[0]
    if "cohort" in user_df.columns:
        user_cohort = user_df["cohort"].iloc[0]
    else:
        user_cohort = None

    blocks = []
    block_counter = 0

    for _, block_subset in user_df.groupby("gap_indicator"):
        # Resample to 15-min
        block_subset = block_subset.copy()
        block_subset['read_at'] = pd.to_datetime(block_subset['read_at']).dt.round('15min')
        
        block_subset = block_subset.set_index("read_at").resample(
                RESAMPLE_INTERVAL,
            ).agg({
            'user_id': 'first',
            'cohort': 'first',
            'val': 'mean',  
            'loc_read_at': 'first',
        })

        # Interpolate glucose values
        block_subset["val"] = block_subset["val"].astype(float).interpolate(method="time")

        # Interpolate local read_at if present
        if "loc_read_at" not in block_subset.columns:
            block_subset["loc_read_at"] = pd.NaT
        block_subset["loc_read_at"] = interpolate_datetime_column(block_subset,
                                                                  "loc_read_at",
                                                                  index_name="read_at")

        block_subset["user_id"] = user_id
        if user_cohort is not None:
            block_subset["cohort"] = user_cohort

        block_subset["timeseries_block_id"] = f"{user_id}__{block_counter}"
        block_counter += 1

        blocks.append(block_subset)

    resampled_user_df = pd.concat(blocks)
    resampled_user_df.reset_index(inplace=True)  # restore 'read_at' as a column
    return resampled_user_df

log.info("Resampling glucose data at 15-minute intervals...")
grouped_users = raw_glucose_df.groupby("user_id")
resampled_glucose_list = []
for _, user_data in tqdm(grouped_users, desc="Resampling CGM user-by-user"):
    resampled_glucose_list.append(resample_cgm_for_user(user_data))

resampled_glucose_df = pd.concat(resampled_glucose_list, ignore_index=True)
log.info(f"Total duplicate read_at timestamps handled: {total_duplicate_count}")

# (Optional) Save resampled CGM to CSV for reference
resampled_cgm_path = os.path.join(OUTPUT_DIRECTORY, f"{FILENAME_PREFIX}glucose_resampled.csv")
resampled_glucose_df.to_csv(resampled_cgm_path, index=False)
log.info(f"Resampled CGM data saved to: {resampled_cgm_path}")

# ---------------------------------------------------------------------------
# 3. AGGREGATE MFR DATA BY DISH_ID
# ---------------------------------------------------------------------------

def merge_food_cluster(dish_dataframe: pd.DataFrame) -> dict:
    """
    Given all MFR rows that belong to a single dish_id, aggregate them into a 
    single representative row. This includes summing up nutritional values 
    and combining textual references.

    Parameters
    ----------
    dish_dataframe : pd.DataFrame
        Slice of the MFR DataFrame filtered by a single dish_id.

    Returns
    -------
    dict
        A dictionary with aggregated nutritional information and references 
        to the original rows.
    """
    
    # Ensure we have only one unique eaten_at timestamp in this cluster
    unique_eaten_at_values = dish_dataframe["eaten_at"].unique()
    assert len(unique_eaten_at_values) == 1, (
        f"Multiple 'eaten_at' times found within dish_id cluster: {unique_eaten_at_values}"
    )

    # Prepare output dictionary
    aggregated_row = {}

    # Basic info
    user_id = dish_dataframe["fay_user_id"].unique()[0]
    aggregated_row["user_id"] = user_id
    aggregated_row["food_intake_row"] = 1  # Indicator for food intake
    

    # Columns expected to have exactly one unique value
    static_value_keys = [
        "eaten_at", "dish_id", "loc_eaten_hour",
        "loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"
    ]
    for col_name in static_value_keys:
        unique_vals = dish_dataframe[col_name].unique()
        assert len(unique_vals) == 1, (
            f"Expected 1 unique value in '{col_name}' but found {len(unique_vals)}."
        )
        aggregated_row[col_name] = unique_vals[0]

    # Nutritional columns
    sum_cols = [
        "eaten_quantity_in_gram", "energy_kcal_eaten", "carb_eaten",
        "fat_eaten", "protein_eaten", "fiber_eaten", "alcohol_eaten"
    ]
    # Fill NaNs with zeros for safe summations 
    ## create a separate copy to avoid pandas warnings
    nutritional_data = dish_dataframe[sum_cols].copy()
    nutritional_data.fillna(0, inplace=True)

    for col_name in sum_cols:
        aggregated_row[col_name] = float(nutritional_data[col_name].sum())

    # One-hot columns for food groups
    all_food_groups = {
        'vegetables_fruits', 'grains_potatoes_pulses', 'unclassified',
        'non_alcoholic_beverages', 'dairy_products_meat_fish_eggs_tofu',
        'sweets_salty_snacks_alcohol', 'oils_fats_nuts'
    }
    present_food_groups = set(dish_dataframe["food_group_cname"].unique())
    for group_name in all_food_groups:
        aggregated_row[group_name] = 1 if group_name in present_food_groups else 0

    # Combine references (these might be row IDs, indexes, etc.)
    original_row_ids = dish_dataframe.iloc[:, 0].tolist()
    aggregated_row["mfr_row_ids"] = ",".join(str(x) for x in original_row_ids)

    # Combine textual references for items
    aggregated_row["food_items"] = " && ".join(
        str(x) for x in dish_dataframe["display_name_en"].tolist()
    )
        
    return aggregated_row

log.info("Aggregating MFR data by dish_id...")
aggregated_rows = []
for dish_id_value, group_df in tqdm(filtered_mfr_data.groupby("dish_id"), desc="Aggregating MFR data by dish_id"):
    aggregated = merge_food_cluster(group_df)
    aggregated_rows.append(aggregated)

aggregated_food_df = pd.DataFrame(aggregated_rows)

# (Optional) Save aggregated food data to CSV for reference
aggregated_food_csv = os.path.join(OUTPUT_DIRECTORY, f"{FILENAME_PREFIX}food_aggregated.csv")
aggregated_food_df.to_csv(aggregated_food_csv, index=False)
log.info(f"Aggregated MFR data saved to: {aggregated_food_csv}")

# ---------------------------------------------------------------------------
# 4. ALIGN FOOD TO CGM TIMESTAMPS (WITHIN 2-HOUR WINDOW)
# ---------------------------------------------------------------------------

def align_food_to_cgm_timestamps(
    glucose_df: pd.DataFrame,
    aggregated_food_data: pd.DataFrame,
    max_offset: pd.Timedelta = FOOD_ALIGNMENT_TIME_WINDOW
) -> pd.DataFrame:
    """
    For each food intake event in 'aggregated_food_data', find the closest CGM 
    timestamp in 'glucose_df' (per user) if the difference is <= max_offset.
    Then store that CGM timestamp as 'aligned_eaten_at'.

    Parameters
    ----------
    glucose_df : pd.DataFrame
        Resampled glucose data with columns ["user_id", "read_at", ...].
    aggregated_food_data : pd.DataFrame
        Aggregated MFR data with columns ["user_id", "dish_id", "eaten_at", ...].
    max_offset : pd.Timedelta
        Maximum allowed time difference for aligning MFR to CGM.

    Returns
    -------
    pd.DataFrame
        The aggregated MFR dataframe extended with "aligned_eaten_at".
        Rows without valid alignment are dropped (NaT alignment).
    """
    # Convert relevant columns to datetime
    glucose_df["read_at"] = pd.to_datetime(glucose_df["read_at"])
    aggregated_food_data["eaten_at"] = pd.to_datetime(aggregated_food_data["eaten_at"])

    aligned_rows_list = []

    # Process user-by-user
    for user_id_val in tqdm(glucose_df["user_id"].unique(), desc="Aligning MFR to CGM"):
        user_glucose_subset = glucose_df[glucose_df["user_id"] == user_id_val]
        user_food_subset = aggregated_food_data[aggregated_food_data["user_id"] == user_id_val]

        if user_glucose_subset.empty or user_food_subset.empty:
            # No data for alignment
            continue

        # Restrict food intakes to the time range covered by CGM data
        valid_time_mask = (
            (user_food_subset["eaten_at"] >= user_glucose_subset["read_at"].min()) &
            (user_food_subset["eaten_at"] <= user_glucose_subset["read_at"].max())
        )
        user_food_subset = user_food_subset[valid_time_mask].copy()
        if user_food_subset.empty:
            continue

        # Find nearest CGM timestamp for each food intake
        aligned_timestamps = []
        for _, food_row in user_food_subset.iterrows():
            food_time = food_row["eaten_at"]
            
            # todo: explore the minima based alignment in a future version
            # Index of row in user_glucose_subset with minimal time difference
            closest_idx = (user_glucose_subset["read_at"] - food_time).abs().idxmin()
            closest_cgm_time = user_glucose_subset.loc[closest_idx, "read_at"]
            time_diff = abs(closest_cgm_time - food_time)

            if time_diff <= max_offset:
                aligned_timestamps.append(closest_cgm_time)
            else:
                aligned_timestamps.append(pd.NaT)

        user_food_subset["aligned_eaten_at"] = aligned_timestamps
        # rename the eaten_at column to original_eaten_at
        user_food_subset.rename(
            columns={"eaten_at":  "original_eaten_at"},
            inplace=True
        )
        user_food_subset.dropna(subset=["aligned_eaten_at"], inplace=True)

        aligned_rows_list.append(user_food_subset)

    aligned_food_df = pd.concat(aligned_rows_list, ignore_index=True) if aligned_rows_list else pd.DataFrame()
    return aligned_food_df

log.info("Aligning aggregated MFR data to CGM timestamps (within 2h)...")
aligned_food_df = align_food_to_cgm_timestamps(
    resampled_glucose_df,
    aggregated_food_df,
    max_offset=FOOD_ALIGNMENT_TIME_WINDOW
)


# ---------------------------------------------------------------------------
# 5. MERGE FOOD CLUSTERS THAT ALIGN TO THE SAME CGM TIMESTAMP
# ---------------------------------------------------------------------------

def merge_food_clusters_same_cgm_timestamp(group: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple dish_ids align to the same CGM timestamp, aggregate them into 
    one row by summing nutritional values, combining references, etc.

    Parameters
    ----------
    group : pd.DataFrame
        Rows that share the same (aligned_eaten_at, user_id).

    Returns
    -------
    pd.DataFrame
        DataFrame of a single row that aggregates the entire group.
    """
    # If only one row, just return it
    if len(group) == 1:
        return group

    # Ensure single aligned_eaten_at
    unique_aligned_times = group["aligned_eaten_at"].unique()
    assert len(unique_aligned_times) == 1, (
        f"Multiple aligned CGM times found in group: {unique_aligned_times}"
    )

    merged_entry = {}
    merged_entry["user_id"] = group["user_id"].unique()[0]
    merged_entry["aligned_eaten_at"] = unique_aligned_times[0]
    merged_entry["food_intake_row"] = 1

    # We can take the mean (or earliest) for numeric time columns
    columns_for_mean = ["original_eaten_at", "loc_eaten_hour"]
    for col_name in columns_for_mean:
        merged_entry[col_name] = group[col_name].mean()

    # For these categorical columns, we can take the first
    columns_for_single_value = ["loc_eaten_dow", "loc_eaten_dow_type", "loc_eaten_season"]
    for col_name in columns_for_single_value:
        merged_entry[col_name] = group[col_name].iloc[0]

    # Combine dish_ids
    merged_entry["dish_id"] = "||".join(str(x) for x in group["dish_id"].unique())

    # Sum nutritional columns
    sum_cols = [
        "eaten_quantity_in_gram", "energy_kcal_eaten", "carb_eaten",
        "fat_eaten", "protein_eaten", "fiber_eaten", "alcohol_eaten"
    ]
    for col_name in sum_cols:
        merged_entry[col_name] = group[col_name].sum()

    # Combine one-hot columns
    all_food_groups = [
        'vegetables_fruits', 'grains_potatoes_pulses', 'unclassified',
        'non_alcoholic_beverages', 'dairy_products_meat_fish_eggs_tofu',
        'sweets_salty_snacks_alcohol', 'oils_fats_nuts'
    ]
    for group_col in all_food_groups:
        merged_entry[group_col] = int(group[group_col].any())

    # Combine references
    merged_entry["mfr_row_ids"] = ",".join(group["mfr_row_ids"].tolist())
    merged_entry["food_items"] = " && ".join(group["food_items"].tolist())

    return pd.DataFrame([merged_entry])

log.info("Merging multiple dish_ids aligned to the same CGM time...")
merged_food_clusters = []
grouped_by_timestamp = aligned_food_df.groupby(["aligned_eaten_at", "user_id"])
for _, group_df in tqdm(grouped_by_timestamp, desc="Merging dish clusters"):
    merged_subset = merge_food_clusters_same_cgm_timestamp(group_df)
    merged_food_clusters.append(merged_subset)

final_aligned_food_df = pd.concat(merged_food_clusters, ignore_index=True)

# ---------------------------------------------------------------------------
# 6. COMBINE (LEFT-JOIN) RESAMPLED CGM WITH ALIGNED FOOD
# ---------------------------------------------------------------------------

log.info("Final merge: CGM + aligned MFR data...")

merged_ppgr_df = resampled_glucose_df.merge(
    final_aligned_food_df,
    how="left",
    left_on=["read_at", "user_id"],
    right_on=["aligned_eaten_at", "user_id"]
)

# Sanity check
assert len(merged_ppgr_df) == len(resampled_glucose_df), (
    "Merged dataset row count differs from original CGM row count."
)

# ---------------------------------------------------------------------------
# 7. Post Process Food Data: CLEANUP / FILL NA / RE-DERIVE LOCAL TIME FEATURES
# ---------------------------------------------------------------------------

# Remove any leftover "Unnamed:..." columns from merges
for drop_col in ["Unnamed: 0", "Unnamed: 0.1"]:
    if drop_col in merged_ppgr_df.columns:
        merged_ppgr_df.drop(columns=[drop_col], inplace=True)

# Default 0 for rows with no food matched
merged_ppgr_df["food_intake_row"] = merged_ppgr_df["food_intake_row"].fillna(0)

nutrient_cols = [
    "eaten_quantity_in_gram", "energy_kcal_eaten", "carb_eaten",
    "fat_eaten", "protein_eaten", "fiber_eaten", "alcohol_eaten"
]
merged_ppgr_df[nutrient_cols] = merged_ppgr_df[nutrient_cols].fillna(0)

# Namespace all nutrient columns with "food__"
nutrient_mapping = {col: f"food__{col}" for col in nutrient_cols}
merged_ppgr_df.rename(columns=nutrient_mapping, inplace=True)

# One-hot columns for food groups
food_group_cols = [
    'vegetables_fruits', 'grains_potatoes_pulses', 'unclassified',
    'non_alcoholic_beverages', 'dairy_products_meat_fish_eggs_tofu',
    'sweets_salty_snacks_alcohol', 'oils_fats_nuts'
]
merged_ppgr_df[food_group_cols] = merged_ppgr_df[food_group_cols].fillna(0)

# Namespace all food group columns with "food__"
food_group_mapping = {col: f"food__{col}" for col in food_group_cols}
merged_ppgr_df.rename(columns=food_group_mapping, inplace=True)

# Ensure food_intake_row is boolean
merged_ppgr_df["food_intake_row"] = merged_ppgr_df["food_intake_row"].astype(bool)

# Parse "loc_read_at" to datetime if present
if "loc_read_at" in merged_ppgr_df.columns:
    merged_ppgr_df["loc_read_at"] = pd.to_datetime(
        merged_ppgr_df["loc_read_at"], format="mixed", errors="coerce"
    )

# If you want to revise your local eaten-hour/dow/season from CGM local time:
if "loc_read_at" in merged_ppgr_df.columns:
    # Re-derive the hour
    merged_ppgr_df["loc_eaten_hour"] = (
        merged_ppgr_df["loc_read_at"].dt.hour +
        merged_ppgr_df["loc_read_at"].dt.minute / 60 +
        merged_ppgr_df["loc_read_at"].dt.second / 3600
    )

    # Re-derive day-of-week
    merged_ppgr_df["loc_eaten_dow"] = merged_ppgr_df["loc_read_at"].dt.day_name().str[:3]

    # Weekend vs weekday
    merged_ppgr_df["loc_eaten_dow_type"] = np.where(
        merged_ppgr_df["loc_read_at"].dt.dayofweek < 5, "weekday", "weekend"
    )

    # Meteorological season
    def get_season(month: int) -> str:
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        else:
            return "Winter"

    merged_ppgr_df["loc_eaten_season"] = merged_ppgr_df["loc_read_at"].dt.month.apply(get_season)

# Add a unique identifier for each row, as the number of rows are not going to change from now on
merged_ppgr_df["ppgr_row_id"] = merged_ppgr_df.index

# ---------------------------------------------------------------------------
# 8. ADD SLEEP DATA
# ---------------------------------------------------------------------------

def add_sleep_data(merged_ppgr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sleep data to the merged PPGR DataFrame by marking whether each CGM 
    'read_at' timestamp falls within a sleep interval recorded in raw_sleep_df.
    
    Parameters
    ----------
    merged_ppgr_df : pd.DataFrame
        The master DataFrame containing CGM data.    
    Returns
    -------
    pd.DataFrame
        The updated merged_ppgr_df including an 'is_sleeping' column.
    """
    log.info("Collecting sleep data from DB...")
    raw_sleep_df = ds.Data.get_sleep()
    
    # Convert sleep timestamps to datetime and filter out invalid rows.
    raw_sleep_df["started_at"] = pd.to_datetime(raw_sleep_df["started_at"])
    raw_sleep_df["ended_at"] = pd.to_datetime(raw_sleep_df["ended_at"])
    raw_sleep_df = raw_sleep_df.dropna(subset=["started_at", "ended_at"])
    
    # Only process users with sleep data.
    users_with_sleep = raw_sleep_df["user_id"].unique()
    ppgr_subset = merged_ppgr_df[merged_ppgr_df["user_id"].isin(users_with_sleep)].copy()
    
    
    # Initialize the new sleep column in both merged_ppgr_df and ppgr_subset with a default value.
    merged_ppgr_df["is_sleeping"] = False
    ppgr_subset["is_sleeping"] = False
    
    # Process each user separately.
    for user_id, user_ppgr in ppgr_subset.groupby("user_id"): #, desc="Adding per-user sleep data":
        # Get and sort sleep intervals for this user.
        user_sleep = raw_sleep_df[raw_sleep_df["user_id"] == user_id].sort_values("started_at")
        
        # Sort the PPGR rows for this user by 'read_at'.
        user_ppgr_sorted = user_ppgr.sort_values("read_at")
        
        # Use merge_asof: for each PPGR row, find the most recent sleep interval
        # where 'started_at' is not after the 'read_at' of the PPGR record.
        merged_user = pd.merge_asof(
            user_ppgr_sorted,
            user_sleep[["started_at", "ended_at"]].sort_values("started_at"),
            left_on="read_at",
            right_on="started_at",
            direction="backward",
            tolerance=pd.Timedelta(RESAMPLE_INTERVAL)
        )
        
        # Determine if the 'read_at' falls within a sleep interval (i.e., between 'started_at' and 'ended_at').
        is_sleeping = (
            (merged_user["read_at"] >= merged_user["started_at"]) & 
            (merged_user["read_at"] <= merged_user["ended_at"])
        ).fillna(False).astype(bool)
        
        # Update the sleep indicator for these rows.
        ppgr_subset.loc[user_ppgr_sorted.index, "is_sleeping"] = is_sleeping.to_numpy()
        log.info(f"User ID: {user_id}, Percentage Sleeping: {100 * is_sleeping.mean():0.2f} %")
    
    # Merge the sleep data back into the master DataFrame.
    merged_ppgr_df.loc[ppgr_subset.index, "is_sleeping"] = ppgr_subset["is_sleeping"].fillna(False)
    
    return merged_ppgr_df


if INCLUDE_SLEEP_DATA:
    merged_ppgr_df = add_sleep_data(merged_ppgr_df)

    percentage_sleeping = 100 * merged_ppgr_df["is_sleeping"].mean() 
    log.info(f"Percentage time participants sleeping in the dataset: {percentage_sleeping:0.2f} %")

    log.info("Added sleep data to the dataset.")
else:
    log.warning("Ignoring Sleep Data")

# ---------------------------------------------------------------------------
# 8. ADD ACTIVITY DATA
# ---------------------------------------------------------------------------

def add_activity_data(merged_ppgr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds objective activity data to the merged PPGR DataFrame by aligning each
    record with the most recent activity interval for that user.
    """
    from utils import get_activity_name_to_intensity_mappings
    
    log.info("Collecting objective physical activity data from DB...")
    activity_df = (
        db.AnalyticsDb()
        .select("""
            SELECT * FROM pa_activity pa
            WHERE start_date NOTNULL AND stop_date NOTNULL
        """)
        .rename(columns={"user_id_fk": "user_id"})
    )
    log.info(f"Collected {len(activity_df)} rows of activity data.")

    # Convert date columns to datetime and drop any rows missing these values
    activity_df['start_date'] = pd.to_datetime(activity_df['start_date'])
    activity_df['stop_date'] = pd.to_datetime(activity_df['stop_date'])
    activity_df = activity_df.dropna(subset=['start_date', 'stop_date'])
    
    # Map activity name to intensity
    # TODO: if the activity mapping changes or is not exhaustive, this will break
    activity_name_to_intensity = get_activity_name_to_intensity_mappings()
    
    activity_df["intensity_inferred"] = activity_df["activity"].apply(lambda x: activity_name_to_intensity[x]) 

    # Only process users with activity data
    users_with_activity = activity_df['user_id'].unique()
    ppgr_subset = merged_ppgr_df[merged_ppgr_df['user_id'].isin(users_with_activity)].copy()

    # Define a mapping: original column from the DB -> (new column, default value)
    activity_mapping = {
        "activity": ("activity__name", "NA"),
        "intensity_inferred": ("activity__intensity_inferred", "NA"),
        "distance": ("activity__distance", 0.0),
        "altitude_gain": ("activity__altitude_gain", 0.0),
        "altitude_loss": ("activity__altitude_loss", 0.0),
        "calories_burned": ("activity__calories_burned", 0.0),
        "average_hr": ("activity__average_heart_rate", 0.0),
        "max_hr": ("activity__max_heart_rate", 0.0)
    }

    # Initialize the new activity columns in merged_ppgr_df + ppgr_subset with their default values.
    for (_, (new_col, default)) in activity_mapping.items():
        merged_ppgr_df[new_col] = default 
        ppgr_subset[new_col] = default

    # Process each user separately.
    for user_id, user_ppgr in tqdm(ppgr_subset.groupby('user_id'), desc="Adding per-user activity data"):
        # Get and sort activity intervals for the user.
        user_activity = activity_df[activity_df['user_id'] == user_id].sort_values('start_date')
        # Sort the PPGR rows by 'read_at'.
        user_ppgr_sorted = user_ppgr.sort_values('read_at')
        
        for activity_row in user_activity.itertuples():
            # Find the PPGR rows that fall within the activity interval.
            within_activity = (user_ppgr_sorted['read_at'] >= activity_row.start_date) & (user_ppgr_sorted['read_at'] <= activity_row.stop_date)
            # Get the actual indices from the sorted subset.
            indices = user_ppgr_sorted.index[within_activity]
            
            # Assign the activity columns using the mapping.
            for orig_col, (new_col, default) in activity_mapping.items():
                try:
                    ppgr_subset.loc[indices, new_col] = getattr(activity_row, orig_col)
                except:
                    import pdb; pdb.set_trace()
            
        # Log the percentage of rows with activity data for the user.
        percent_activity = 100 * (ppgr_subset["activity__name"].fillna("NA") != "NA").mean()
        log.info(f"User ID: {user_id}, Percentage Activity: {percent_activity:0.2f} %")

    # After processing, update the master DataFrame.
    for (_, (new_col, default)) in activity_mapping.items():
        merged_ppgr_df.loc[ppgr_subset.index, new_col] = ppgr_subset[new_col].fillna(default)

    return merged_ppgr_df

if INCLUDE_ACTIVITY_DATA:
    merged_ppgr_df = add_activity_data(merged_ppgr_df)
    percent_activity = 100 * (merged_ppgr_df["activity__name"].fillna("NA") != "NA").mean()
    log.info(f"Percentage of rows with activity data: {percent_activity:0.2f} %")
else:
    log.warning("Ignoring Activity Data")


# ---------------------------------------------------------------------------
# 11. Add user demographics data
# ---------------------------------------------------------------------------

def gather_users_demographics_data(merged_ppgr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds user demographics data to the merged PPGR DataFrame by merging with the users table.
    """
    log.info("Collecting user demographics data from DB...")
    users_info_raw_df = db.FayDb().select("""
    SELECT * FROM users
    """)
    users_questionnaires_df = db.FayDb().select("""
        SELECT * FROM questionnaires
    """)
    users_df = users_info_raw_df[users_info_raw_df["id"].isin(merged_ppgr_df.user_id)] # Filter users to only include participants in the merged PPGR DataFrame
    users_df = users_df[["id", "age", "gender", "weight", "height", "bmi", "microbiome_tube", "wake_up", "go_sleep", "birthdate"]] # NOTE: birthdate will be dropped later for privacy reasons
    log.info(f"Collected {len(users_df)} rows of users data.")
    
    # recalculate bmi
    users_df['bmi'] = users_df['weight'] / ((users_df['height'] / 100) ** 2)
    
    # Calculate age from birthdate for users who don't have an age (about 17 in the current dataset)
    ### collect the time range of the current study from the mean year of the user data in merged_ppgr_df
    study_start_year = merged_ppgr_df["read_at"].dt.year.min()
    # For users who don't have an age, calculate the age based on the study start year
    users_df.loc[users_df["age"].isna(), "age"] = (study_start_year - users_df.loc[users_df["age"].isna(), "birthdate"].dt.year)
    
    # Drop the birthdate column
    users_df = users_df.drop(columns=["birthdate"])

    log.debug(f'age_unavailable: {users_df["age"].isna().sum()}')
    log.debug(f'gender_unavailable: {users_df["gender"].isna().sum()}')
    log.debug(f'weight_unavailable: {users_df["weight"].isna().sum()}')
    log.debug(f'height_unavailable: {users_df["height"].isna().sum()}')
    log.debug(f'bmi_unavailable: {users_df["bmi"].isna().sum()}')
    log.debug(f'microbiome_tube_unavailable: {users_df["microbiome_tube"].isna().sum()}')
    log.debug(f'wake_up_unavailable: {users_df["wake_up"].isna().sum()}')
    
    
    # Extract the relevant columns from the questionnaires DataFrame.
    questionnaire_columns = [
        "user_id", 
        "edu_degree", 
        "income", 
        "household_desc", 
        "job_status", 
        "smoking", 
        "general_hunger_level", 
        "morning_hunger_level", 
        "mid_hunger_level", 
        "evening_hunger_level", 
        "health_state", 
        "physical_activities_frequency"
    ]

    # Create a copy with only the desired columns.
    questionnaire_df = users_questionnaires_df[questionnaire_columns].copy()

    # Rename all columns (except 'user_id') to be namespaced under "user__"
    questionnaire_df = questionnaire_df.rename(
        columns=lambda col: col if col == "user_id" else f"user__{col}"
    )

    # If your primary user DataFrame uses a different key such as "id", rename it to "user_id" first.
    users_df = users_df.rename(columns={"id": "user_id"})
    
    # Rename all columns (except 'user_id') to be namespaced under "user__"
    users_df = users_df.rename(
        columns=lambda col: col if col == "user_id" else f"user__{col}"
    )

    # Now join (merge) both DataFrames on the 'user_id' column.
    merged_users_df = pd.merge(users_df, questionnaire_df, on="user_id", how="left")

    return merged_users_df


users_demographics_df = gather_users_demographics_data(merged_ppgr_df)

# ---------------------------------------------------------------------------
# 10. ADD GRANULAR DISH DATA 
# ---------------------------------------------------------------------------

def gather_dishes_data(merged_ppgr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gathers the data about all the dishes represented in the dataset, and returns a DataFrame.
    """
    global filtered_mfr_data
    
    # List of all dishes in the dataset
    all_dishes = merged_ppgr_df["dish_id"].dropna().unique()
    all_dishes = [str(x).split("||") for x in all_dishes] # some dishes have been merged into a single row, unmerge them
    # flatten dishes
    all_dishes = [item for sublist in all_dishes for item in sublist]
    # convert to int
    all_dishes = [int(x) for x in all_dishes] # this is needed for the next filtering step
    
    # Collect only the subset of the MFR data that is represented in the dataset
    dishes_df = filtered_mfr_data[filtered_mfr_data["dish_id"].isin(all_dishes)]
    
    dish_columns_of_interest = [
        "food_id",
        "dish_id",
        "eaten_quantity_in_gram",
        "display_name_en",
        "energy_kcal_eaten",
        "carb_eaten",
        "fat_eaten",
        "protein_eaten",
        "fiber_eaten",
        "alcohol_eaten",
        "sugar_eaten",
        "dairy_products_meat_fish_eggs_tofu",
        "vegetables_fruits",
        "sweets_salty_snacks_alcohol",
        "non_alcoholic_beverages",
        "grains_potatoes_pulses",
        "oils_fats_nuts",
        "fay_user_id",
        "cohort",
        "loc_eaten_hour",
        "loc_eaten_on",
        "loc_eaten_dow",
        "loc_eaten_dow_type",
        "loc_eaten_season"
        
    ]
    
    # Drop the Unnamed: 0 column to keep things clean 
    dishes_df = dishes_df.drop(columns=["Unnamed: 0"])
    
    # Rename the columns to be namespaced under "dish__"
    dishes_df = dishes_df.rename(
        columns=lambda col: col if col in ["dish_id", "food_id"] else f"food__{col}"
    )

    return dishes_df.reset_index(drop=True)

dishes_df = gather_dishes_data(merged_ppgr_df)

# ---------------------------------------------------------------------------
# 11. Add Microbiome Data 
# ---------------------------------------------------------------------------

def gather_microbiome_data(merged_ppgr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gathers the microbiome data for the users in the merged PPGR DataFrame.
    """
    from utils import preprocess_microbiome_data_with_RA_and_CLR
    log.info("Collecting and pre-processing microbiome data")
    
    global users_demographics_df # This has the microbiome tube id that we can use to correlate the microbiome data
    global MICROBIOME_DATA_PATH
    global MICROBIOME_MIN_RELATIVE_ABUNDANCE
    global MICROBIOME_MIN_PRESENCE_FRACTION
    
    # Load the microbiome data
    user_microbiome = pd.read_csv(MICROBIOME_DATA_PATH, sep="\t", skiprows=[0])

    
    # Call the pre-processing function 
    processed_df = preprocess_microbiome_data_with_RA_and_CLR(
        user_microbiome,
            transpose_data=True,
            min_relative_abundance=MICROBIOME_MIN_RELATIVE_ABUNDANCE,    # OTU must reach 5% RA
            min_presence_fraction=MICROBIOME_MIN_PRESENCE_FRACTION,     # in at least 5% of samples (adjust to 0.10 if desired)
            scale_data=True,
            plot_heatmap=False
    )
    
    # Calculate the mean microbiome data for all the users (to use for users with no microbiome data)
    mean_microbiome_row = processed_df.mean(axis=0)
    
    # Map the processed microbiome data to the users in the merged PPGR DataFrame
    _users_demographics_df = users_demographics_df.copy()
    _users_demographics_df["user__microbiome_tube"] = _users_demographics_df["user__microbiome_tube"].astype(int)
    
    # Calculate map of sample id to user id
    sample_id_to_user_id = {}
    for index, row in _users_demographics_df.iterrows():
        sample_id_to_user_id[f"sample{row['user__microbiome_tube']}"] = row["user_id"]
    
    # identify samples for which we have the data
    def get_user_from_sample_id(sample_id):
        _user_df = _users_demographics_df[_users_demographics_df["user__microbiome_tube"] == float(sample_id.replace("sample", ""))]
        if len(_user_df) == 0:
            return None
        else:
            return int(_user_df["user_id"].iloc[0])

    # Calculate the corresponding user ids for each sample
    user_ids = []
    for sampled_id in processed_df.index.tolist():
        user_id = get_user_from_sample_id(sampled_id) # this returns None if the sample id is not found in the user demographics df
        user_ids.append(user_id)
    
    # Add the user ids to the processed microbiome data
    processed_df["user_id"] = processed_df.index.map(get_user_from_sample_id)
    
    # Calculate the number of samples for which user was not found
    num_samples_with_no_user_id = processed_df["user_id"].isna().sum()
    log.info(f"Found {num_samples_with_no_user_id} samples for which user was not found. Ignoring them.")
    
    # drop rows where user_id is None
    relevant_microbiome_df = processed_df[processed_df["user_id"].notna()].reset_index().set_index("user_id").drop(columns=["index"])
    relevant_microbiome_df.index = relevant_microbiome_df.index.astype(int)

    # Reset the index to be able to add the user_id easily for the users with no microbiome data
    relevant_microbiome_df = relevant_microbiome_df.reset_index()
        
    # Check for users with no microbiome data
    user_ids_not_in_microbiome_df = set(merged_ppgr_df["user_id"].unique().tolist()) - set(relevant_microbiome_df["user_id"].tolist())
    
    log.info(f"Found {len(user_ids_not_in_microbiome_df)} users with no microbiome data. Adding the mean microbiome data for these users.")
    
    
    # Add the mean microbiome data for the users with no microbiome data    
    for user_id in user_ids_not_in_microbiome_df:
        # Create a temporary DataFrame with the mean row
        if user_id is not None:
            mean_microbiome_df = pd.DataFrame([mean_microbiome_row])
            mean_microbiome_df["user_id"] = user_id
            # Concatenate with the existing DataFrame
            relevant_microbiome_df = pd.concat([relevant_microbiome_df, mean_microbiome_df])
    
    # Double check that all users have microbiome data
    assert sorted(relevant_microbiome_df["user_id"].tolist()) == sorted(merged_ppgr_df["user_id"].unique().tolist())
        
    return relevant_microbiome_df


user_microbiome_df = gather_microbiome_data(merged_ppgr_df)

# --------------------------------------------------------------------------- 
# 12. Add Food Embeddings
# ---------------------------------------------------------------------------


def gather_food_embeddings(merged_ppgr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gathers the food embeddings for the dishes in the merged PPGR DataFrame.
    """
    global dishes_df
    log.info("Collecting food embeddings for the food items in the dishes data")
    
    # Load the food embeddings
    log.info(f"Loading food embeddings from {FOOD_EMBEDDINGS_PATH}")
    food_embeddings_df = pd.read_pickle(FOOD_EMBEDDINGS_PATH)


    # Check that all food_ids in dishes_df are present in food_embeddings_df
    missing_food_ids = set(dishes_df["food_id"]) - set(food_embeddings_df["food_id"])
    if missing_food_ids:
        raise ValueError(f"The following food_ids are missing from the food embeddings data: {missing_food_ids}. Cannot proceed.")
    else:
        log.info(f"All food_ids in dishes_df are present in food_embeddings_df")

    log.info(f"Filtering food embeddings to only include the food items in the dishes data")
    food_embeddings_df = food_embeddings_df[food_embeddings_df["food_id"].isin(dishes_df["food_id"])]
    
    # Set the food_id as the index and sort the index
    food_embeddings_df.set_index("food_id", inplace=True)
    food_embeddings_df.sort_index(inplace=True)
    return food_embeddings_df


food_embeddings_df = gather_food_embeddings(merged_ppgr_df)

# ---------------------------------------------------------------------------
# 12. EXPORT FINAL DATASET
# ---------------------------------------------------------------------------

output_merged_path = os.path.join(
    OUTPUT_DIRECTORY,
    f"{FILENAME_PREFIX}fay-ppgr-processed-and-aggregated-{VERSION}.csv"
)
merged_ppgr_df.to_csv(output_merged_path, index=False)
log.info(f"Final merged dataset saved to: {output_merged_path}")

# Write the users demographics data to a CSV file
users_demographics_df.to_csv(os.path.join(OUTPUT_DIRECTORY, f"{FILENAME_PREFIX}users-demographics-data-{VERSION}.csv"), index=False)
log.info(f"Users demographics data saved to: {os.path.join(OUTPUT_DIRECTORY, f'{FILENAME_PREFIX}users-demographics-data-{VERSION}.csv')}")

# Gather the data about all the dishes represented in the dataset, and save it to a CSV file
dishes_df.to_csv(os.path.join(OUTPUT_DIRECTORY, f"{FILENAME_PREFIX}dishes-data-{VERSION}.csv"), index=False)
log.info(f"Dishes data saved to: {os.path.join(OUTPUT_DIRECTORY, f'{FILENAME_PREFIX}dishes-data-{VERSION}.csv')}")

# Gather the microbiome data, and save it to a CSV file
user_microbiome_df.to_csv(os.path.join(OUTPUT_DIRECTORY, f"{FILENAME_PREFIX}microbiome-data-{VERSION}.csv"), index=False)
log.info(f"Microbiome data saved to: {os.path.join(OUTPUT_DIRECTORY, f'{FILENAME_PREFIX}microbiome-data-{VERSION}.csv')}")

# Gather the food embeddings, and save it to a CSV file
food_embeddings_pickle_path = os.path.join(
    OUTPUT_DIRECTORY, f"{FILENAME_PREFIX}food-embeddings-{VERSION}.pkl.gz"
)
food_embeddings_df.to_pickle(
    food_embeddings_pickle_path,
    compression='gzip'
)
log.info(f"Food embeddings saved to: {food_embeddings_pickle_path}")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------

log.info("Data processing pipeline completed successfully.")
