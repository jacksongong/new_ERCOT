from pathlib import Path
import argparse

import pandas as pd
from tqdm import tqdm
from loguru import logger

class DataCombiner:
    def __init__(
        self,
        data_root: str | Path,
        save_csv_root: str | Path
    ):
        self.data_root = Path(data_root)
        self.save_csv_root = Path(save_csv_root)

        self.da_price_root = data_root / "unzip_NP4-180-ER"
        self.rt_price_root = data_root / "unzip_NP6-785-ER"
        self.wind_production_root = data_root / "unzip_NP4-732-CD"
        self.solar_production_root = data_root / "unzip_NP4-737-CD"
        self.wind_production_region_root = data_root / "unzip_NP4-742-CD"
        self.solar_production_region_root = data_root / "unzip_NP4-745-CN"
        self.load_forecast_seven_day_root = data_root / "unzip_NP3-565"

    def _combine_da_price(self):
        df = []
        for file in tqdm(
            sorted(self.da_price_root.iterdir()),
            desc="Reading DA price files"
        ):
            df.extend(pd.read_excel(file, sheet_name=None).values())
        df: pd.DataFrame = pd.concat([v for v in df if not v.isna().all(axis=None)])
        logger.info("Combining DA Prices")
        start_time = df["Delivery Date"] + " " + df["Hour Ending"].apply(lambda k: f"{int(k[:2])-1:02}" + ":00")
        start_time = pd.DatetimeIndex(start_time)
        utc_start_time = start_time.tz_localize("US/Central", ambiguous=df["Repeated Hour Flag"] == "N").tz_convert("UTC")
        index = pd.MultiIndex.from_arrays([utc_start_time, df["Settlement Point"]], names=["utc_start_time", None])
        assert any(index.duplicated()) is False
        s = pd.Series(df["Settlement Point Price"].values, index=index)
        df = s.unstack(level=-1).drop(columns=["HB_BUSAVG", "HB_HUBAVG"])
        self.save_csv_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.save_csv_root / "da_price.csv")

    def _combine_rt_price(self):
        df = []
        for file in tqdm(
            sorted(self.rt_price_root.iterdir()),
            desc="Reading RT price files"
        ):
            df.extend(pd.read_excel(file, sheet_name=None).values())
        df: pd.DataFrame = pd.concat(v for v in df if not v.isna().all(axis=None))
        logger.info("Combining RT Prices")
        df = df[
            (~ df["Settlement Point Name"].isin(["HB_BUSAVG", "HB_HUBAVG"]))
            &
            (df["Settlement Point Type"] != "LZEW")
        ].drop(columns=["Settlement Point Type"]).dropna()
        start_time = df["Delivery Date"] + " " + (df["Delivery Hour"] - 1).astype(str).str.zfill(2) + ":00"
        start_time = pd.DatetimeIndex(start_time)
        utc_start_time = start_time.tz_localize(
            "US/Central",
            ambiguous=df["Repeated Hour Flag"] == "N"
        ).tz_convert("UTC")
        df["utc_start_time"] = utc_start_time
        df = df.groupby(["utc_start_time", "Settlement Point Name"])["Settlement Point Price"].mean().unstack(level=-1)
        self.save_csv_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.save_csv_root / "rt_price.csv")

    @staticmethod
    def _load_single_np3_565_cd_csv(path) -> pd.DataFrame:
        """The length of the returned DataFrame might be 191, 192, or 193 because of daylight savings
        """
        df = pd.read_csv(path)
        df = df[df.InUseFlag == "Y"].copy()
        start_time = df.DeliveryDate + " " + df.HourEnding.apply(lambda k:str(int(k.split(":")[0])-1).zfill(2)) + ":00"
        start_time = pd.DatetimeIndex(start_time)
        utc_start_time = start_time.tz_localize("US/Central", ambiguous=df.DSTFlag).tz_convert("UTC")
        df["utc_start_time"] = utc_start_time
        df = df.drop(columns=["DeliveryDate", "HourEnding", "DSTFlag", "InUseFlag"]).set_index("utc_start_time")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="The root path to the unzipped dirs"
    )
    parser.add_argument(
        "--save_csv_root",
        type=Path,
        required=True,
        help="The path to which the csv files will be saved"
    )
    args = parser.parse_args()

    combiner = DataCombiner(args.data_root, args.save_csv_root)
    # combiner._combine_da_price()
    # combiner._combine_rt_price()