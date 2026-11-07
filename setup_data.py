import argparse
import os
import subprocess
from huggingface_hub import snapshot_download


EVAL_REVISION = "f31717fe522c2a6725ca0024980f51169a084ca1"


def download_eval_data():
    snapshot_download(
        repo_id="open-unlearning/eval",
        revision=EVAL_REVISION,
        allow_patterns=[
            f"tofu_{model}_{suffix}/TOFU_EVAL.json"
            for model in ("Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct")
            for suffix in (
                "retain90", "retain95", "retain99",
                "full/evals_forget10", "full/evals_forget05", "full/evals_forget01",
            )
        ],
        repo_type="dataset",
        local_dir="saves/eval",
    )


def download_idk_data():
    snapshot_download(
        repo_id="open-unlearning/idk",
        allow_patterns="*.jsonl",
        repo_type="dataset",
        local_dir="data",
    )


def download_wmdp():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["wget", url, "-O", zip_path], check=True)
    subprocess.run(["unzip", "-P", "wmdpcorpora", zip_path, "-d", dest_dir], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download and setup evaluation data.")
    parser.add_argument(
        "--eval_logs",
        action="store_true",
        help="Download TOFU reference evaluation logs into saves/eval",
    )
    parser.add_argument(
        "--idk",
        action="store_true",
        help="Download idk dataset from HF hub and stores it data/idk.jsonl",
    )
    parser.add_argument(
        "--wmdp",
        action="store_true",
        help="Download and unzip WMDP dataset into data/wmdp",
    )

    args = parser.parse_args()

    if args.eval_logs:
        download_eval_data()
    if args.idk:
        download_idk_data()
    if args.wmdp:
        download_wmdp()


if __name__ == "__main__":
    main()
