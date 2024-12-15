import scipy
from pathlib import Path
import json


def json_to_list(path: Path) -> list:
    results: list = []
    with open(path, "r") as file:
        data = json.load(file)
    for t in data:
        results.append(t.get("passed"))
    return list(map(int, results))


def main():
    our_result: Path = Path("./SmolLM-360M_14_12_2024_results_jl.json")
    copilot_result: Path = Path("./copilot_multiple_predictions.json")

    our_result_data: list = json_to_list(path=our_result)
    copilot_result_data: list = json_to_list(path=copilot_result)

    if len(our_result_data) != len(copilot_result_data):
        print("Error length of the two series is different!")
        return

    # TODO get statistics
    wilcoxon = scipy.stats.wilcoxon(x=our_result_data, y=copilot_result_data)

    print(wilcoxon)
    if wilcoxon.pvalue < 0.05:
        print("Different")


if __name__ == '__main__':
    main()

# results
# SmolLM-135M_12_12_2024_doc -> WilcoxonResult(statistic=0.0, pvalue=2.2571768119076647e-19) -> Different
# SmolLM-135M_13_12_2024     -> WilcoxonResult(statistic=0.0, pvalue=3.744097384202872e-19)  -> Different
# SmolLM-360M_13_12_2024_doc -> WilcoxonResult(statistic=0.0, pvalue=1.7095795875991002e-18) -> Different
# SmolLM-360M_14_12_2024     -> WilcoxonResult(statistic=0.0, pvalue=6.210993425425189e-19)  -> Different
