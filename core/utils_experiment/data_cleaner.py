from pathlib import Path
import argparse
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger


"""If more cleaning methods are added in this file, the corresponding part in data fetcher should be changed.
"""

def clean(
    data_root: Path
):
    # first issue: corrupted 737 report on Jun 30 and Jul 1 of 2022, use 745 to replace some of those corrupted reports
    # ref1: https://www.ercot.com/services/comm/mkt_notices/M-D070122-01
    # ref2: https://www.ercot.com/services/comm/mkt_notices/M-D070122-02
    # corrupted report range:
    # from: cdr.00013483.0000000000000000.20220630.155523.PVGRHRLYAVGACTNP4737.csv
    # to:   cdr.00013483.0000000000000000.20220701.165522.PVGRHRLYAVGACTNP4737.csv
    solar_737_root = data_root / "unzip_NP4-737-CD"
    solor_745_root = data_root / "unzip_NP4-745-CD"
    logger.info("Issue 1: corrupted files of 737 from Jun 30 to Jul 1 of 2022")
    hour_range_start = datetime(2022, 6, 30, 15)
    hour_range_end = datetime(2022, 7, 1, 17)
    mapping = dict() # which file should replace which file
    for file in sorted(solar_737_root.iterdir()):
        date, time = file.stem.split(".")[3:5]
        report_time = datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
        report_hour = datetime.strptime(f"{date} {time[:2]}0000", "%Y%m%d %H%M%S")
        if report_time > report_hour:
            ready_time = report_hour + timedelta(hours=1)
        else:
            ready_time = report_hour
        if hour_range_start < report_time < hour_range_end:
            mapping[ready_time] = file
    # use 745 to replace some of them
    for file in sorted(solor_745_root.iterdir()):
        date, time = file.stem.split(".")[3:5]
        report_time = datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
        report_hour = datetime.strptime(f"{date} {time[:2]}0000", "%Y%m%d %H%M%S")
        if report_time > report_hour:
            ready_time = report_hour + timedelta(hours=1)
        else:
            ready_time = report_hour
        if hour_range_start < report_time < hour_range_end:
            df = pd.read_csv(file)
            df = df[["DELIVERY_DATE", "HOUR_ENDING", "GEN_SYSTEM_WIDE", "COP_HSL_SYSTEM_WIDE", "STPPF_SYSTEM_WIDE", "PVGRPP_SYSTEM_WIDE", "DSTFlag"]]
            df = df.rename(columns={"GEN_SYSTEM_WIDE": "ACTUAL_SYSTEM_WIDE"})
            path_to_replace: Path = mapping[ready_time]
            file_save_path = path_to_replace.with_stem(path_to_replace.stem + "_recover745")
            df.to_csv(file_save_path, index=False)
            logger.info(f"Added file {file_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Handle specific errors in the raw data, should be used before building the dataset for training. ATTENTION: this script added files into the raw data folder!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="The root path to the data fold, should contain subfolders like `unzip_NP4-737-CD` etc"
    )
    args = parser.parse_args()

    clean(args.data_root)
