import os

INPUT_DATA_DIR = "../input_data"
PROCESSED_DATA_DIR = "../processed_data"
PLOTS_DIR = "../plots"

SURVEY_INPUT_DATA_DIR = os.path.join(
    INPUT_DATA_DIR, "survey_monkey_assault_weapons_apr_2024/CSV/"
)

SURVEY_INPUT_DATA_PATH = os.path.join(SURVEY_INPUT_DATA_DIR, "Gun Policy Survey.csv")
SURVEY_CODED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "Gun_Policy_Survey_coded.csv")
SURVEY_METADATA_PATH = os.path.join(
    PROCESSED_DATA_DIR, "Gun_Policy_Survey_metadata.json"
)

COLLECTOR_ID = 430645356.0
