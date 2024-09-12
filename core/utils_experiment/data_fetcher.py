from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger


class Fetcher:
    START_YEAR = 2017
    # we will use the data from 2018 to 2023, 6 years in total
    # but some of the data of the start of 2018 should be fetched from the end of 2017, so here start year is 2017
    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)
        self.mapping = self._preprocess()

    def _preprocess(self):
        # build the mapping of report and its available time
        # for example, a report published 9:30, its available time will be the 10:00 right after it
        # handle situations where multiple reports might be available during one hour
        raise NotImplementedError

    def get(self, *args, **kwargs):
        # get the data in DataFrame given the date
        raise NotImplementedError


class Fetcher565(Fetcher):
    """Fetcher for 565: Seven-Day Load Forecast by Model and Weather Zone
    """
    def _preprocess(self):
        mapping: dict[datetime, Path] = dict() # report_ready_time -> report_file_path
        for file in sorted(self.data_root.iterdir()):
            date, time = file.stem.split(".")[3:5]
            report_time = datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
            report_hour = datetime.strptime(f"{date} {time[:2]}0000", "%Y%m%d %H%M%S")
            if report_time.year < self.START_YEAR:
                continue
            if report_time > report_hour:
                ready_time = report_hour + timedelta(hours=1)
            else:
                ready_time = report_hour
            if ready_time in mapping:
                existed = mapping[ready_time]
                if "xhr" in existed.stem and "xhr" not in file.stem:
                    continue
                elif "xhr" in file.stem and "xhr" not in existed.stem:
                    mapping[ready_time] = file
                else:
                    logger.warning(f"weird: report ready time {ready_time} already in mapping, old {mapping[ready_time]}, new {file}")
                    mapping[ready_time] = file
            else:
                mapping[ready_time] = file
        return mapping

    def get(self, report_time: datetime, target_hour: int = None):
        # return day ahead load forecast, data for either 24 hours or 1 hour
        file = self.mapping[report_time]
        df = pd.read_csv(file)
        df = df[df.InUseFlag == "Y"]
        df.drop(columns=["InUseFlag"], inplace=True)
        target_date = (report_time + timedelta(days=1)).date().strftime("%m/%d/%Y")
        filtered = df[df.DeliveryDate == target_date].copy()
        filtered["HourStarting"] = filtered.HourEnding.map(lambda t: int(t.split(":")[0]) - 1)
        filtered.drop(columns=["HourEnding", "DeliveryDate", "Model"], inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        if len(filtered) < 24:
            filtered.drop(columns=["DSTFlag"], inplace=True)
            assert len(filtered) == 23
            # find the missing hour
            missing_hour = set(range(24))
            for row_i, row in filtered.iterrows():
                missing_hour.remove(int(row["HourStarting"]))
            assert len(missing_hour) == 1
            missing_hour = missing_hour.pop()
            to_add = dict(HourStarting=missing_hour)
            for column in filtered.columns:
                if column not in ["HourStarting"]:
                    to_add[column] = np.nan
            filtered = pd.concat([filtered, pd.DataFrame([to_add])])
            filtered = filtered.sort_values(by="HourStarting")
        elif len(filtered) > 24:
            filtered = filtered[filtered["DSTFlag"] == "N"]
            filtered.drop(columns=["DSTFlag"], inplace=True)
        else:
            filtered.drop(columns=["DSTFlag"], inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        filtered.interpolate(inplace=True)
        assert len(filtered) == 24
        if target_hour is not None:
            filtered = filtered[filtered.HourStarting == target_hour]
            assert len(filtered) == 1, f"{target_hour=} is invalid"

        return filtered


class Fetcher523(Fetcher):
    def _preprocess(self):
        mapping = dict() # report date -> report_file_path
        for file in sorted(self.data_root.iterdir()):
            date = file.stem.split(".")[3]
            report_date = datetime.strptime(date, "%Y%m%d")
            if report_date.year < self.START_YEAR:
                continue
            assert report_date not in mapping
            mapping[report_date.date()] = file
        return mapping

    def get(self, report_date: datetime, target_hour: int = None):
        # return day ahead system lambda, data for 24 hours or 1 hour
        file = self.mapping[report_date.date()]
        df = pd.read_csv(file)
        df = df[df["DSTFlag"] != "Y"]
        df["HourStarting"] = df.HourEnding.map(lambda t: int(t.split(":")[0]) - 1)
        df.drop(columns=["DeliveryDate", "HourEnding", "DSTFlag"], inplace=True)
        if len(df) != 24:
            assert len(df) == 23
            missing_hour = set(range(24))
            for row_i, row in df.iterrows():
                missing_hour.remove(int(row["HourStarting"]))
            assert len(missing_hour) == 1
            missing_hour = missing_hour.pop()
            system_lambda = df.loc[df["HourStarting"].isin([missing_hour - 1, missing_hour + 1]), ["SystemLambda"]].mean().values[0]
            df = pd.concat([df, pd.DataFrame([{"SystemLambda": system_lambda, "HourStarting": missing_hour}])])
            df.sort_values(by="HourStarting", inplace=True)
            df.reset_index(drop=True, inplace=True)
        if target_hour is not None:
            df = df[df["HourStarting"] == target_hour]
            assert len(df) == 1, f"{target_hour=} is invalid"
        return df


class Fetcher732(Fetcher):
    def _preprocess(self):
        mapping: dict[datetime, Path] = dict() # report_ready_time -> report_file_path
        for file in sorted(self.data_root.iterdir()):
            if "retry" in file.stem:
                date, time = file.stem.split("_")[-2:]
            else:
                date, time = file.stem.split(".")[3:5]
            report_time = datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
            report_hour = datetime.strptime(f"{date} {time[:2]}0000", "%Y%m%d %H%M%S")
            if report_time.year < self.START_YEAR:
                continue
            if report_time > report_hour:
                ready_time = report_hour + timedelta(hours=1)
            else:
                ready_time = report_hour
            if ready_time in mapping:
                existed = mapping[ready_time]
                if "xhr" in existed.stem and "xhr" not in file.stem:
                    continue
                elif "xhr" in file.stem and "xhr" not in existed.stem:
                    mapping[ready_time] = file
                else:
                    logger.warning(f"weird, two reports in the same hour, old {existed.stem}, new {file.stem}")
                    mapping[ready_time] = file
            else:
                mapping[ready_time] = file
        return mapping

    def get(self, report_time: datetime, target_hour: int = None):
        # return day ahead wind forecast, date for 24 hours or 1 hour
        file = self.mapping[report_time]
        df = pd.read_csv(file)
        df["HourStarting"] = df["HOUR_ENDING"] - 1
        next_date = report_time.date() + timedelta(days=1)
        df = df[df.DELIVERY_DATE == next_date.strftime("%m/%d/%Y")]
        actual_columns = [c for c in df.columns if c.startswith("ACTUAL_")]
        df = df[df["DSTFlag"] != "Y"]
        df.drop(columns=["HOUR_ENDING", "DSTFlag", "DELIVERY_DATE", *actual_columns], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) != 24:
            assert len(df) == 23
            missing_hour = set(range(24))
            for row_i, row in df.iterrows():
                missing_hour.remove(int(row["HourStarting"]))
            assert len(missing_hour) == 1
            missing_hour = missing_hour.pop()
            dict_to_add = dict(HourStarting=missing_hour)
            for column in df.columns:
                if column in ["HourStarting"]:
                    continue
                dict_to_add[column] = np.nan
            df = pd.concat([df, pd.DataFrame([dict_to_add])])
            df.sort_values(by="HourStarting", inplace=True)
            df.reset_index(drop=True, inplace=True)
        df.interpolate(inplace=True)
        if target_hour is not None:
            df = df[df["HourStarting"] == target_hour]
            assert len(df) == 1, f"{target_hour=} is invalid"
        return df


class Fetcher737(Fetcher):
    def _preprocess(self):
        mapping: dict[datetime, Path] = dict() # report_ready_time -> report_file_path
        for file in sorted(self.data_root.iterdir()):
            date, time = file.stem.split(".")[3:5]
            report_time = datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
            report_hour = datetime.strptime(f"{date} {time[:2]}0000", "%Y%m%d %H%M%S")
            if report_time.year < self.START_YEAR:
                continue
            if report_time > report_hour:
                ready_time = report_hour + timedelta(hours=1)
            else:
                ready_time = report_hour
            if ready_time in mapping:
                existed = mapping[ready_time]
                if "xhr" in existed.stem and "xhr" not in file.stem:
                    continue
                elif "xhr" in file.stem and "xhr" not in existed.stem:
                    mapping[ready_time] = file
                elif "recover745" in file.stem and "recover745" not in existed.stem:
                    mapping[ready_time] = file
                else:
                    logger.warning(f"weird, two reports in the same hour, old {existed.stem}, new {file.stem}")
                    mapping[ready_time] = file
            else:
                mapping[ready_time] = file
        return mapping

    def get(self, report_time: datetime, target_hour: int = None):
        # return day ahead wind forecast, date for 24 hours or 1 hour
        file = self.mapping[report_time]
        df = pd.read_csv(file)
        df["HourStarting"] = df["HOUR_ENDING"] - 1
        next_date = report_time.date() + timedelta(days=1)
        df = df[df.DELIVERY_DATE == next_date.strftime("%m/%d/%Y")]
        df = df[df.DSTFlag != "Y"]
        df.drop(columns=["HOUR_ENDING", "DSTFlag", "ACTUAL_SYSTEM_WIDE", "DELIVERY_DATE"], inplace=True)
        if len(df) != 24:
            assert len(df) == 23
            missing_hour = set(range(24))
            for row_i, row in df.iterrows():
                missing_hour.remove(int(row["HourStarting"]))
            assert len(missing_hour) == 1
            missing_hour = missing_hour.pop()
            dict_to_add = dict(HourStarting=missing_hour)
            for column in df.columns:
                if column in ["HourStarting"]:
                    continue
                dict_to_add[column] = np.nan
            df = pd.concat([df, pd.DataFrame([dict_to_add])])
            df.sort_values(by="HourStarting", inplace=True)
            df.reset_index(drop=True, inplace=True)
        df.interpolate(inplace=True)
        if target_hour is not None:
            df = df[df.HourStarting == target_hour]
            df.reset_index(drop=True, inplace=True)
            assert len(df) == 1, f"{target_hour=} is invalid"
        return df
