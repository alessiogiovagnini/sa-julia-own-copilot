import pandas as pd
from pandas import DataFrame


def main():
    df: DataFrame = pd.read_csv("./merged_julia_final.csv")
    # total entries before: 996531
    # number without duplicate: 993397
    # number without empty bodies: 981373
    # function signature have no NaN
    df = df.drop_duplicates()
    df = df[df["function_body"].notna()]

    print(df.columns)
    print(f"number of entries: {df.shape[0]}")

    print(f"num of docstring: {df[df["docstring"].notna()].shape[0]}")
    # total docstring (not nan):    186071

    df.to_csv(index=False, path_or_buf="./filtered_final.csv")


if __name__ == '__main__':
    main()
