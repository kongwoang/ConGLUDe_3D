import argparse
import logging
import os
from pathlib import Path
import zipfile

import requests


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def download_file(url: str, output_path: Path, timeout: int = 60) -> None:
    logging.info("Start downloading: %s", output_path.name)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    logging.info(
                        "Downloading %s: %.1f%% (%d/%d bytes)",
                        output_path.name,
                        percent,
                        downloaded,
                        total_size,
                    )
                else:
                    logging.info(
                        "Downloading %s: %d bytes downloaded",
                        output_path.name,
                        downloaded,
                    )

    logging.info("Finished downloading: %s", output_path)


def extract_zip(zip_path: Path, extract_folder: Path) -> None:
    logging.info("Start extracting: %s -> %s", zip_path.name, extract_folder)
    extract_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        logging.info("Zip contains %d files/folders", len(members))
        z.extractall(extract_folder)

    logging.info("Finished extracting: %s", extract_folder)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Download evaluation datasets from Zenodo and extract them."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to download (default: download all datasets).",
    )
    args = parser.parse_args()

    record_id = 18933183
    metadata_url = f"https://zenodo.org/api/records/{record_id}"
    output_dir = Path("data/datasets/test_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Fetching Zenodo metadata from: %s", metadata_url)
    try:
        response = requests.get(metadata_url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logging.exception("Failed to fetch Zenodo metadata: %s", e)
        raise SystemExit(1)

    files = data.get("files", [])
    logging.info("Found %d files in Zenodo record %s", len(files), record_id)

    matched = 0

    for file_info in files:
        download_url = file_info["links"]["self"]
        filename = file_info["key"]
        dataset_name = filename[:-4] if filename.endswith(".zip") else filename

        if args.dataset_name is not None and dataset_name != args.dataset_name:
            logging.debug("Skipping dataset: %s", dataset_name)
            continue

        matched += 1
        zip_path = output_dir / filename
        extract_folder = output_dir / dataset_name

        logging.info("Processing dataset: %s", dataset_name)
        logging.info("Download URL: %s", download_url)

        try:
            download_file(download_url, zip_path)
            extract_zip(zip_path, extract_folder)

            logging.info("Removing archive: %s", zip_path)
            zip_path.unlink()

            logging.info("Dataset ready: %s", dataset_name)
        except requests.RequestException as e:
            logging.exception("Download failed for %s: %s", dataset_name, e)
        except zipfile.BadZipFile as e:
            logging.exception("Invalid zip file for %s: %s", dataset_name, e)
        except OSError as e:
            logging.exception("Filesystem error while processing %s: %s", dataset_name, e)

    if matched == 0:
        if args.dataset_name is None:
            logging.warning("No datasets found in the Zenodo record.")
        else:
            logging.warning("Dataset '%s' was not found in the Zenodo record.", args.dataset_name)
    else:
        logging.info("Done. Processed %d dataset(s).", matched)


if __name__ == "__main__":
    main()