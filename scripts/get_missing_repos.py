import pandas
from pandas import DataFrame
import os
from pathlib import Path
import re


all_extracted_repos_names: list[str] = []


def filter_row(full_name: str) -> bool:
    name: str = full_name.split("/").pop()
    name = name.removesuffix(".jl")
    return name not in all_extracted_repos_names


def main():
    csv_output_path: Path = Path("./csv_output")
    julia_repos_path: Path = Path("./julia_repo_list.csv")

    files_list: list[str] = os.listdir(csv_output_path)

    for file in files_list:
        original_name = re.sub("-\\d+.csv", "", file)
        all_extracted_repos_names.append(original_name)

    df: DataFrame = pandas.read_csv(julia_repos_path)

    filtered_df = df[df["name"].apply(func=filter_row)]

    # print(df["name"].size)
    # print(filtered_df["name"].size)
    # print(df["name"].size - filtered_df["name"].size)
    # print(len(files_list))

    print(filtered_df["name"])


if __name__ == '__main__':
    main()
