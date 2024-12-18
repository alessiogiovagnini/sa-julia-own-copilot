import os
import pandas as pd
import sh
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading
from julia_function_extractor import start_extraction


def multi_thread_analysis(repo_names: list[str], max_threads: int = 16) -> None:
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        try:
            executor.map(start_analysis_on_repo, repo_names, timeout=600)
        except TimeoutError as e:
            print("######## Timeout")
            print(e)
    print("####### Program finished! ########")


def start_analysis_on_repo(repo_name: str):
    full_path: str = f"https://github.com/{repo_name}"
    current_thread_id: int = threading.get_ident()
    tmp_repo_dir: Path = Path("./repos")
    tmp_repo_dir.mkdir(exist_ok=True)
    tmp_destination: Path = Path(tmp_repo_dir, f"tmp-{Path(repo_name).stem}-{current_thread_id}")

    analyze_repo(url=full_path, destination=tmp_destination, repo_name=repo_name)


def analyze_repo(url: str, destination: Path, repo_name: str) -> None:
    """
    clone a repository in the destination folder and analyze its content
    :param repo_name: name of the repository, like: author/repo-name
    :param url: the url of the repository, like https://github.com/author/repo-name
    :param destination: path where the repo is cloned, the folder target will be created if it not exists,
        and it will be DELETED at the end with all the repository
    """
    destination.mkdir(exist_ok=True)
    try:
        sh.git.clone(url, destination)

    except Exception as e:
        print(f"####### Failed to clone repository {url}")
        print(e)
        destination.rmdir()

    if len(os.listdir(destination)) == 0:
        destination.rmdir()
        return

    csv_output_dir: Path = Path("./csv_output")
    csv_output_dir.mkdir(exist_ok=True)

    current_thread_id: int = threading.get_ident()
    simple_repo_name: str = Path(repo_name).stem
    csv_output_file: Path = Path(csv_output_dir, f"{simple_repo_name}-{current_thread_id}.csv")

    print(f"###### Start reading {destination}")

    for root, dirs, files in os.walk(destination):
        for file in files:
            if file.endswith(".jl"):
                file_path: Path = Path(root, file)
                size = os.path.getsize(file_path)
                if size < 1024 * 1024:
                    try:
                        start_extraction(repo_name=repo_name, file_path=file_path, output_file=csv_output_file)
                    except Exception as e:
                        print(f"######## Failed to read file {file_path}")
                        print(e)

    shutil.rmtree(destination)
    destination.rmdir()

    print(f"####### End reading {destination}")
    return


def start_script(csv_input: Path):
    df = pd.read_csv(csv_input)
    all_repos_names: list[str] = df["name"].to_list()
    multi_thread_analysis(repo_names=all_repos_names, max_threads=10)


if __name__ == '__main__':
    csv_path: Path = Path("./julia_repo_list.csv")
    start_script(csv_input=csv_path)
