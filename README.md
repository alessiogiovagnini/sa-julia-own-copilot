# sa-julia-own-copilot
second assignment for Software Analytics - 2024

# Installing
creating virtual env
```shell
 python3.12 -m venv .venv
```
activating it
```shell
source .venv/bin/activate
```
check if is correct
```shell
which python
```
installing requirements
```shell
pip install -r requirements.txt
```

# Running the scripts
Downloading all the data from the repository:
```shell
cd scripts
python clone-repo.py
```
This will generate a list of csv file in the folder `csv_output`,
then to merge the files in one:
```shell
python merge_csv.py
```

# Other scripts
`remove_benchmark_functions.py` is used to remove the benchmark functions
from the dataset, you need to edit the paths inside the script.

`statistics.py` is used to calculate the statistics from the json files
resulting from the `evaluate.sh` script, you also need to edit the paths
inside the script.