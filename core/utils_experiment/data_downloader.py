from pathlib import Path
from datetime import datetime, timedelta
import json
import zipfile
from time import sleep
import argparse

import yaml
import requests
from loguru import logger
from tqdm import tqdm
from tenacity import retry, wait_fixed


class DataDownloader:
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    ARCHIVE_URL = "https://api.ercot.com/api/public-reports/archive/"
    def __init__(
        self,
        save_root: Path | str,
        credential_path: Path | str = None,
        sleep_second: float = 0.3,
        reuse_cache: bool = False,
        incremental: bool = False,
    ):
        self.save_root = Path(save_root)
        self.sleep_second = sleep_second
        self.reuse_cache = reuse_cache
        self.incremental = incremental
        if credential_path is None:
            credential_path = Path(__file__).parent / "info.yaml"
        self.credential_path = credential_path
        # The credential yaml should look like this:
        # username: your_username to API explorer
        # password: your_password to API explorer
        # Ocp-Apim-Subscription-Key: your_subscription_key to API explorer, after you subscribe to ERCOT public API, you will see this in your profile in API explorer
        self.token_expire_time = datetime.now()
        self._authenticate()

    def _authenticate(self):
        if self.token_expire_time > datetime.now():
            return
        with self.credential_path.open(mode="r", encoding="utf8") as f:
            info = yaml.safe_load(f)
        data = {
            "grant_type": "password",
            "username": info["username"],
            "password": info["password"],
            "response_type": "id_token",
            "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
            "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70"
        }
        response = requests.post(self.AUTH_URL, data)
        self.id_token = response.json()["id_token"]
        self.subscription_key = info["Ocp-Apim-Subscription-Key"]
        self.token_expire_time = datetime.now() + timedelta(minutes=30)

    @retry(wait=wait_fixed(2))
    def _get(self, url, **params):
        self._authenticate()
        headers = {
            'Content-Type':'application/json',
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Authorization': f'Bearer {self.id_token}',
        }
        r = requests.get(
            url,
            params=params,
            headers=headers,
        )
        sleep(self.sleep_second)
        if r.status_code == 200:
            return r
        else:
            raise Exception(f"Error: {r.status_code}, {r.content}")

    def _get_task_list(self, emil_id: str):
        save_json_path = self.save_root / f"{emil_id}.json"
        if save_json_path.is_file():
            if not self.reuse_cache:
                logger.debug(f"{save_json_path} exists and not reusing cache, discarding the json file, re-fetch the list")
                save_json_path.unlink()
            else:
                logger.debug(f"{save_json_path} exists, reusing it")
                return
        r = self._get(self.ARCHIVE_URL+emil_id, page=1).json()
        total_pages = r["_meta"]["totalPages"]
        report_type_id = str(r["product"]["reportTypeId"])
        logger.info(f"Total pages for {report_type_id} ({emil_id}): {total_pages}")
        tasks = []
        for page in tqdm(range(1, total_pages+1), desc=f"Fetch doc list for {report_type_id} ({emil_id})"):
            r = self._get(self.ARCHIVE_URL+emil_id, page=page).json()
            for archive in r["archives"]:
                tasks.append(
                    (
                        archive["docId"],
                        archive["postDatetime"]
                    )
                )
            if self.incremental and tasks:
                doc_id = tasks[-1][0]
                zipfile_path = self.save_root / emil_id / f"{doc_id}.zip"
                if zipfile_path.is_file():
                    logger.debug(f"incremental download enabled, the oldest report in this page is found, stop fetching the list")
                    break
        self.save_root.mkdir(exist_ok=True, parents=True)
        with open(save_json_path, "w", encoding="utf8") as f:
            json.dump(
                {
                    "report_type_id": report_type_id,
                    "emil_id": emil_id,
                    "tasks": tasks
                },
                f,
            )

    def _download_files(self, emil_id: str) -> list[Path]:
        with open(
            self.save_root / f"{emil_id}.json",
            "r",
            encoding="utf8"
        ) as f:
            data = json.load(f)
        tasks = data["tasks"]
        emil_id = data["emil_id"]
        save_root = self.save_root / emil_id
        save_root.mkdir(exist_ok=True, parents=True)
        need_to_download = []
        for doc_id, post_datetime in tasks:
            zip_path = save_root / f"{doc_id}.zip"
            if zip_path.is_file():
                continue
            need_to_download.append((doc_id, post_datetime))
        print(f"Total doc to download for {emil_id}: {len(need_to_download)}")
        downloaded_zips: list[Path] = []
        for doc_id, post_datetime in tqdm(need_to_download[::-1], desc=f"Download {emil_id}"):
            zip_path = save_root / f"{doc_id}.zip"
            r = self._get(
                self.ARCHIVE_URL+emil_id,
                download=doc_id
            )
            with open(zip_path, "wb") as f:
                f.write(r.content)
            downloaded_zips.append(zip_path)
        return downloaded_zips

    def _unzip_all(self, emil_id: str, downloaded_zips: list[Path] = None):
        unzip_root = self.save_root / f"unzip_{emil_id}"
        unzip_root.mkdir(exist_ok=True, parents=True)
        zip_root = self.save_root / emil_id
        if downloaded_zips is not None:
            zip_files = downloaded_zips
        else:
            zip_files = list(zip_root.glob("*.zip"))
        for zip_path in tqdm(zip_files, desc=f"Unzip {emil_id}"):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_root)

    def download_one_emil(self, emil_id: str):
        self._get_task_list(emil_id)
        downloaded_zips = self._download_files(emil_id)
        if self.incremental:
            self._unzip_all(emil_id, downloaded_zips)
        else:
            self._unzip_all(emil_id)

if __name__ == "__main__":
    EMIL_ID_LIST = [
        "NP3-565-CD",
        "NP4-523-CD",
        "NP4-732-CD",
        "NP4-737-CD",
        "NP4-742-CD",
        "NP4-745-CD",
    ]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--save_root",
        type=Path,
        required=True,
        help="The root path to the outputs"
    )
    parser.add_argument(
        "--credential_path",
        type=Path,
        default=None,
        help="The path to the yaml file that contains the username, password, and Ocp-Apim-Subscription-Key for API usage. If not given, use info.yaml in the api_utils.py current folder."
    )
    parser.add_argument(
        "--sleep_second",
        type=float,
        default=0.3,
        help="How long it should sleep after each API call"
    )
    parser.add_argument(
        "--emil_id",
        type=str,
        nargs="+",
        default=EMIL_ID_LIST,
        help="The emil_id of the products to download"
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="If given and there is a JSON cache file for the EMIL, it will not fetch the latest report list. This flag should be used for resuming the downloading."
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="If given, will check the last archive in each page, if the last item exists, it will stop fetching the next page. This speeds up the downloading process, but it assumes the download is in the order of OLD -> New. This flag should be used when running daily data fetching."
    )
    args = parser.parse_args()
    downloader = DataDownloader(
        save_root=args.save_root,
        credential_path=args.credential_path,
        sleep_second=args.sleep_second,
        reuse_cache=args.reuse_cache,
        incremental=args.incremental,
    )
    for emil_id in args.emil_id:
        downloader.download_one_emil(emil_id)
