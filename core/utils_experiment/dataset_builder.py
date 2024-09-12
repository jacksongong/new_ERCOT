from pathlib import Path
from datetime import datetime, timedelta
import argparse

import pandas as pd

from data_fetcher import Fetcher565, Fetcher523, Fetcher732, Fetcher737


def column_rename(df: pd.DataFrame, prefix: str):
    mapping = dict()
    for column in df.columns:
        mapping[column] = prefix + "_" + column
    return df.rename(columns=mapping)


def main(data_root: Path, output_csv_path: Path):
    load_fetcher = Fetcher565(data_root / "unzip_NP3-565-CD")
    system_lambda_fetcher = Fetcher523(data_root / "unzip_NP4-523-CD")
    wind_fetcher = Fetcher732(data_root / "unzip_NP4-732-CD")
    solar_fetcher = Fetcher737(data_root / "unzip_NP4-737-CD")

    # we need the dataset contains the load, supply and price info from Jan. 1, 2018 to Dec. 31, 2023.
    START_TIME = datetime(2017, 12, 31, 10)
    END_TIME = datetime(2023, 12, 31, 10)
    df_list = []
    d = START_TIME
    while d != END_TIME:
        load = column_rename(load_fetcher.get(d).set_index("HourStarting"), "load")
        system_lambda = system_lambda_fetcher.get(d).set_index("HourStarting")
        wind = column_rename(wind_fetcher.get(d).set_index("HourStarting"), "wind")
        solar = column_rename(solar_fetcher.get(d).set_index("HourStarting"), "solar")
        df = pd.concat([load, system_lambda, wind, solar], axis=1)
        df["timestamp"] = df.index.map(lambda t: (d + timedelta(days=1)).strftime("%m/%d/%Y") + f" {t:02}:00")
        df = df.set_index("timestamp", drop=True)
        df_list.append(df)
        d += timedelta(days=1)

    pd.concat(df_list, axis=0).to_csv(output_csv_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the DAM dataset from 2018 to 2023 with load, wind, solar predictions and the system lambda.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="The path to the data root, should contain subdir like `unzip_NP4-732-CD` etc"
    )
    parser.add_argument(
        "--output_csv_path",
        type=Path,
        required=True,
        help="The path to save the dataset csv file"
    )
    args = parser.parse_args()

    main(args.data_root, args.output_csv_path)
