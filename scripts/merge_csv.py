import os
from pathlib import Path


def main():
    csv_dir: Path = Path("./csv_output")

    files_list: list[str] = os.listdir(csv_dir)

    if len(files_list) < 2:
        print(f"Need 2 or more files, found: {len(files_list)}")
        return

    first_file: str = files_list.pop()
    first_file_path: Path = Path(csv_dir, first_file)

    final_output_file: Path = Path("./merged_julia_final.csv")

    if final_output_file.exists():
        print("File already exist")
        return

    with open(final_output_file, "ab") as out_file:
        with open(first_file_path, "rb") as first:
            out_file.writelines(first)  # also write the header

        for current_file in files_list:
            current_file_path: Path = Path(csv_dir, current_file)
            with open(current_file_path, "rb") as current:
                next(current)  # skip the header
                out_file.writelines(current)

    print("Finished")


if __name__ == '__main__':
    main()
