import pandas as pd
import pickle
import os
import re
import argparse
import pathlib

def seperate_df(pickle_path, path_24, pickle_24, csv_file_path, pickle_path_updated):
    common_delivery = 'DeliveryDate'
    common_time = 'Hour_Ending'

    with open(pickle_path, 'rb') as file:
        load_final_df = pickle.load(file)

    print(load_final_df)
    DST_Dates = [['03/11/2018', '11/04/2018'], ['03/10/2019', '11/03/2019'], ['03/08/2020', '11/01/2020'],
                 ['03/14/2021', '11/07/2021'], ['03/13/2022', '11/06/2022'], ['03/12/2023', '11/05/2023'],
                 ['03/10/2024', '11/03/2024']]

    na_row_col_indices = [(index, load_final_df.columns.get_loc(col)) for col in load_final_df.columns for index in load_final_df[col][load_final_df[col].isna()].index]
    for location in na_row_col_indices:
        row_index = location[0]
        column_index = location[1]
        empty_cell_above = load_final_df.iloc[row_index + 1, column_index]
        empty_cell_below = load_final_df.iloc[row_index - 1, column_index]


        if pd.isna(empty_cell_above) and not pd.isna(empty_cell_below):
            load_final_df.iloc[row_index, column_index] = empty_cell_below

        elif not pd.isna(empty_cell_above) and pd.isna(empty_cell_below):
            load_final_df.iloc[row_index, column_index] = empty_cell_above

        elif not pd.isna(empty_cell_above) and not pd.isna(empty_cell_below):
            load_final_df.iloc[row_index, column_index] = (empty_cell_above + empty_cell_below) / 2

    lower_bound = '1:00'
    upper_bound = '4:00'
    condition_lower = '2:00'
    condition_upper = '3:00'

    for i in range(0, len(DST_Dates)):


        lower_bound_df = load_final_df.loc[(load_final_df[common_delivery] == DST_Dates[i][0]) & (load_final_df[common_time] == condition_lower)].copy()
        upper_bound_df = load_final_df.loc[(load_final_df[common_delivery] == DST_Dates[i][0]) & (load_final_df[common_time] == condition_upper)].copy()

        lower_bound_df_exsisting = load_final_df.loc[(load_final_df[common_delivery] == DST_Dates[i][0]) & (load_final_df[common_time] == lower_bound)].copy()
        upper_bound_df_exsisting = load_final_df.loc[(load_final_df[common_delivery] == DST_Dates[i][0]) & (load_final_df[common_time] == upper_bound)].copy()

        if lower_bound_df.empty and upper_bound_df.empty:

            lower_bound_df = lower_bound_df_exsisting

            upper_bound_df = upper_bound_df_exsisting

            upper_bound_df[common_time] = condition_upper

            lower_bound_df[common_time] = condition_lower

            load_final_df = pd.concat([load_final_df, lower_bound_df, upper_bound_df]).sort_values(by = common_time).reset_index(drop = True)  #this will concat two rows for each iteration of the for loop

    df_hour_list = []
    for i in range(0, 24):
        time = f'{i}:00'
        filterd_hour_individually = load_final_df[load_final_df[common_time] == time]
        filterd_hour_individually[common_delivery] = pd.to_datetime(filterd_hour_individually[common_delivery])
        filterd_hour_individually = filterd_hour_individually.sort_values(by=common_delivery).reset_index(drop=True)
        df_hour_list.append(filterd_hour_individually)

    print(df_hour_list)

    for i in range(0, len(df_hour_list)):
        individual_path_csv = os.path.join(path_24, f"{i}_Hour_Df.csv")
        df_hour_list[i].to_csv(individual_path_csv, index=True)

        individual_path_pkl = os.path.join(pickle_24, f'{i}_Hour_Pickle_Data.pkl')
        with open(individual_path_pkl, 'wb') as file:
            pickle.dump(df_hour_list[i], file)




    load_final_df['DateTime'] = pd.to_datetime(load_final_df[common_delivery] + ' ' + load_final_df[common_time])

    load_final_df = load_final_df.sort_values(by = 'DateTime').reset_index(drop=True)

    load_final_df = load_final_df.drop(columns = ['DateTime'] )

    load_final_df.to_csv(csv_file_path, index=True)
    with open(pickle_path_updated, 'wb') as file:
        pickle.dump(load_final_df, file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Final_Data_Proccessor INPUTS: Please run the python script from the parent directory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--created_directory",
        type=pathlib.Path,
        required=True,
        help="Absolute path to all files that you want to store-- PLEASE TYPE THE SAME PATH for '--created_directory' AS in FINAL_DATA_PROCCESSSER.PY Script"
    )

    args = parser.parse_args()
    created_directory = args.created_directory

    ercot_data_path = created_directory.joinpath('ERCOT_Data')
    ercot_data_path.mkdir(exist_ok=True)

    pickle_files_path = ercot_data_path.joinpath('Csv_Files')
    pickle_files_path.mkdir(exist_ok=True)


    pickle_files_path = ercot_data_path.joinpath('Pickle_Successful')
    pickle_files_path.mkdir(exist_ok=True)

    pickle_path = os.path.join(ercot_data_path, 'Final_Pickle_Data.pkl')
    pickle_path_updated = os.path.join(ercot_data_path, 'Final_Pickle_Data_DST_Filtered.pkl')



    path_24 = ercot_data_path.joinpath('24_Hour_Data_Csv')
    path_24.mkdir(exist_ok=True)

    pickle_24 = ercot_data_path.joinpath('24_Hour_Pickle')
    pickle_24.mkdir(exist_ok=True)

    csv_file_path = os.path.join(ercot_data_path, 'Final_Concated_DF_DST_Filtered.csv')


    seperate_df(pickle_path, path_24, pickle_24, csv_file_path, pickle_path_updated)