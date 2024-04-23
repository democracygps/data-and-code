# assault weapons 2024 democracyGPS data and models

Goal is to predict support for 4 laws in all US congressional districts.

# Steps to repeat analysis published at XXX

## set up environment:
cd models
### create python virtual environment
python -m venv ./venv-assault-weapons-2024
### activate python virtual environment
source ./venv-assault-weapons-2024/bin/activate
### install needed packages
pip install -r requirements

## preprocess house data
python preprocess_house_district_data.py

## preprocess survey data
python preprocess_assault_weapon_data.py

## run simulations for each house and senate district
python ./run_models.py ../processed_data/house_district_data_baseline.csv
python ./run_models.py ../processed_data/house_district_data_right_only.csv
python ./run_models.py ../processed_data/house_district_data_without_center.csv

## plot results
python ./merge_and_plot.py

## export data to excel
python export_data_to_excel.py

# Data sources:
https://www.rand.org/content/dam/rand/pubs/tools/TL300/TL354/RAND_TL354.database.zip
https://www.pewresearch.org/religion/religious-landscape-study/compare/party-affiliation/by/state#party-affiliation
https://raw.githubusercontent.com/fivethirtyeight/data/master/partisan-lean/fivethirtyeight_partisan_lean_DISTRICTS.csv
https://raw.githubusercontent.com/fivethirtyeight/data/master/gop-delegate-benchmarks-2024/delegate_targets.csv

Private Survey Monkey survey

###
Note: The original code was developed in the private repo https://github.com/crkrenn/democracyGPS-models