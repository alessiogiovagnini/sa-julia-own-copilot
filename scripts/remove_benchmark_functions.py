from pathlib import Path
import pandas as pd
from pandas import DataFrame
import re


def main():
    path: Path = Path("/Users/alessiogiovagnini/Desktop/Master/software_analytics/assignment2/sa-julia-own-copilot/benchmark/humaneval-jl-reworded.jsonl")

    jsonObj = pd.read_json(path_or_buf=path, lines=True)

    fun = []

    for a in jsonObj["prompt"].to_list():
        res = re.search("function \w+\(", a)

        fun_name = res.group()[8:]
        fun_name = fun_name[:-1]

        fun.append(fun_name)

    def helper(row):
        for sub_str in fun:
            if sub_str in row["function_signature"]:
                return False
        return True

    df: DataFrame = pd.read_csv("./merged_julia_final.csv")

    filtered_df = df[df.apply(helper, axis=1)]

    filtered_df.to_csv("./merged_julia_final_filtered.csv", index=False)

    pass


if __name__ == '__main__':
    main()
