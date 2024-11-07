import sh
from pathlib import Path
import shutil


def analyze_repo(url: str, destination: Path) -> None:
    """
    clone a repository in the destination folder and analyze its content
    :param url: the url of the repository, like https://github.com/author/repo-name
    :param destination: path where the repo is cloned, the folder target will be created if it not exist
        and it will be DELETED at the end with all the repository
    """
    destination.mkdir(exist_ok=True)
    sh.git.clone(url, destination)

    # TODO get every Julia file inside this repo and for each parse all the functions and save in the csv

    shutil.rmtree(destination)


if __name__ == '__main__':
    test: str = "https://github.com/anirban166/rsa-cryptosystem"
    dest: Path = Path("../tmp")
    dest.mkdir(exist_ok=True)

    analyze_repo(test, dest)





