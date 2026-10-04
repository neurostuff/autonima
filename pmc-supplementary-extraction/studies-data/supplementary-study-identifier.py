import pandas as pd
import numpy as np
import json

#==============================================#
#             GLOBAL VARIABLES                 #
#==============================================#
COORDINATE_FILE_NAME = 'coordinate_parsing_results.json'
FULLTEXT_FILE_NAME = 'fulltext_retrieval_results.json'
FILE_PATH = 'C:\\Users\\antho\\Neuroinformatics-Lab\\pmc-supplementary-downloads\\studies-data\\'


#==============================================#
#                  FUNCTIONS                   #
#==============================================#

#-------------------#
#     LOAD DATA     #
#-------------------#                 

# Extract info from coordinate parsing json files
def load_coordinate_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    coordinate_parsed_pmids = []
    for study in data['studies']:
        coordinate_parsed_pmids = [study['pmid'] for study in data['studies']]

    return coordinate_parsed_pmids

# Extract info from full-text retrieval json files
def load_retrieval_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    df_retrieval = pd.DataFrame(data['studies_with_fulltext'])[
        ['pmid', 'pmcid', 'fulltext_available', 'coordinates_found']
    ]

    df_retrieval['pmcid'] = df_retrieval['pmcid'].astype('Int64')

    return df_retrieval

# Load included studies file
def load_included_studies(file_path):
    df_included = pd.read_csv(file_path)
    return df_included

#---------------------#
#    DATA LOGIC       #
#---------------------#

def data_logic(coordinate_parsed_pmids, df_retrieval, df_included):
    # Filter out null PMCID values to identify pubget studies
    df = df_retrieval[df_retrieval['pmcid'].notnull()]

    # Filter out True fulltext_available & False coordinates_found indicating coordinates weren't found in the full text
    df_pubget = df[(df['fulltext_available'] == True) & (df['coordinates_found'] == False)]

    # Cross reference with included_studies.csv to see if any of the studies that were retrieved from pubget are in the included_studies.csv file
    #df_pubget_included = pd.merge(df_included, df_pubget, left_on = 'study_pmid', right_on = 'pmid', how = 'inner')

    # Take difference between pubget included studies and coordinate parsed studies
    df_final = df_pubget[~df_pubget['pmid'].isin(coordinate_parsed_pmids)]

    #df_final = df_final.drop(columns = ['study_pmid', 'doi'])

    return df_final

#==============================#
#           MAIN               #
#==============================#

if __name__ == "__main__":
    project_names = ['cue-reactivity', 'decision-making', 'dementia', 'emotion-regulation', 'executive-function', 
                   'problem-solving', 'social', 'vbm-ptsd', 'vbm-substance-use']

    # Create a list of dfs for all PMIDs to look into
    all_final = []

    for name in project_names:
        # Load all data
        coordinates_parsed_pmids = load_coordinate_json(FILE_PATH + name + "\\" + COORDINATE_FILE_NAME)
        df_retrieval = load_retrieval_json(FILE_PATH + name + "\\" + FULLTEXT_FILE_NAME)
        df_included = load_included_studies(FILE_PATH + "included_studies.csv")
        df_final = data_logic(coordinates_parsed_pmids, df_retrieval, df_included)

        all_final.append(df_final)

        # Output to csv for each project
        df_final.to_csv(FILE_PATH + name + "\\" + name + "_pmids.csv", index = False)

    df_all_final = pd.concat(all_final, ignore_index = True)

    # Output complete list of all PMIDs across all projects
    df_all_final.to_csv(FILE_PATH + "\\" + "all-pmids.csv", index = False)