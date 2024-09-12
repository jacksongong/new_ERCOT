import pandas as pd
import pickle
import os
import argparse
import pathlib
import re
# file_path_523 = '/media/jacksong/Data/electricity_price_prediction/Electricity_Prediction_Scripts/Combined_Data/Pickled_Successful/_523_Final_Df.pkl'
# with open(file_path_523, 'rb') as file:
#     load_523 = pickle.load(file)
#
# file_path_565 = '/media/jacksong/Data/electricity_price_prediction/Electricity_Prediction_Scripts/Combined_Data/Pickled_Successful/_565_Final_Df.pkl'
# with open(file_path_565, 'rb') as file:
#     load_565 = pickle.load(file)
#
# file_path_732 = '/media/jacksong/Data/electricity_price_prediction/Electricity_Prediction_Scripts/Combined_Data/Pickled_Successful/_732_Final_Df.pkl'
# with open(file_path_732, 'rb') as file:
#     load_732 = pickle.load(file)
#
# file_path_737 = '/media/jacksong/Data/electricity_price_prediction/Electricity_Prediction_Scripts/Combined_Data/Pickled_Successful/_737_Final_Df.pkl'
# with open(file_path_737, 'rb') as file:
#     load_737 = pickle.load(file)
#
# file_path_742 = '/media/jacksong/Data/electricity_price_prediction/Electricity_Prediction_Scripts/Combined_Data/Pickled_Successful/_742_Final_Df.pkl'
# with open(file_path_742, 'rb') as file:
#     load_742 = pickle.load(file)


#______________________________________________________________
# file_path_523 = 'F:/electricity_price_prediction/Electricity_Prediction_Scripts/No_Historical_Combined/Pickled_Successful/_523_Final_Df.pkl'
#     with open(file_path_523, 'rb') as file:
#         load_523 = pickle.load(file)
#
#     file_path_565 = 'F:/electricity_price_prediction/Electricity_Prediction_Scripts/No_Historical_Combined/Pickled_Successful/_565_Final_Df.pkl'
#     with open(file_path_565, 'rb') as file:
#         load_565 = pickle.load(file)
#
#     file_path_732 = 'F:/electricity_price_prediction/Electricity_Prediction_Scripts/No_Historical_Combined/Pickled_Successful/_732_Final_Df.pkl'
#     with open(file_path_732, 'rb') as file:
#         load_732 = pickle.load(file)
#
#     file_path_737 = 'F:/electricity_price_prediction/Electricity_Prediction_Scripts/No_Historical_Combined/Pickled_Successful/_737_Final_Df.pkl'
#     with open(file_path_737, 'rb') as file:
#         load_737 = pickle.load(file)
#
#     file_path_742 = 'F:/electricity_price_prediction/Electricity_Prediction_Scripts/No_Historical_Combined/Pickled_Successful/_742_Final_Df.pkl'
#     with open(file_path_742, 'rb') as file:
#         load_742 = pickle.load(file)

def concat_2D(list_load, csv_path, pickle_path):
    list_loaded_data = list_load
    load_523 = list_loaded_data[0]
    load_565 = list_loaded_data[1]
    load_732 = list_loaded_data[2]
    load_737 = list_loaded_data[3]
    load_742 = list_loaded_data[4]


    columns_to_rename_523 = list_loaded_data[0].columns[2:]
    new_column_names_523 = {old: old + '_ActualValues' for old in columns_to_rename_523}
    load_523.rename(columns = new_column_names_523, inplace=True)

    columns_to_rename_565 = list_loaded_data[1].columns[2:]
    new_column_names_565  = {old: old + '_WeatherZone' for old in columns_to_rename_565}
    load_565.rename(columns = new_column_names_565 , inplace=True)

    columns_to_rename_732 = list_loaded_data[2].columns[2:]
    new_column_names_732  = {old: old + '_WindPowerProduction' for old in columns_to_rename_732}
    load_732.rename(columns = new_column_names_732 , inplace=True)

    columns_to_rename_737 = list_loaded_data[3].columns[2:]
    new_column_names_737  = {old: old + '_SolarPowerProduction' for old in columns_to_rename_737}
    load_737.rename(columns = new_column_names_737 , inplace=True)

    columns_to_rename_742 = list_loaded_data[4].columns[2:]
    new_column_names_742 = {old: old + '_WindPowerProductionGeographical' for old in columns_to_rename_742}
    load_742.rename(columns = new_column_names_742, inplace=True)

    common_time = 'Hour_Ending'
    common_delivery = 'DeliveryDate'

    def format_hour(hour):
        return f"{hour}:00"

    def check_format(x):
        x = str(x)
        return bool(re.match(r'^\d{1,2}:00$', x))

    reformatted_list = []
    for item in list_loaded_data:
        boolean_column = item[common_time].apply(check_format)
        if boolean_column.all():
            item[common_time] = item[common_time]
        else:
            item[common_time] = item[common_time].astype(str)
            item[common_time] = item[common_time].apply(format_hour)
        reformatted_list.append(item)

    final_horizontal_concat = []
    for y in range(18, 25):  # 18,25
        for m in range(1, 13):  # 1,13
            every_other_updated = 1
            for every_other_updated in range(1, 32):  # 1,32
                hour = 0
                for hour in range (0,24):
                    y_cushion = f'{y:02}'
                    m_cushion = f'{m:02}'
                    d_cushion = f'{every_other_updated:02}'


                    format_time = f'{str(hour)}:00'
                    format_date = f'{m_cushion}/{d_cushion}/20{y_cushion}'

                    format_time_1 = f'{str(hour)}:00:00:00'
                    format_date_1 = f'{m_cushion}/{d_cushion}/20{y_cushion}'

                    final_df_columns = []
                    print(format_date_1)


                    for item in reformatted_list:
                        iteration_df_columns = []


                        desired_delivery = item[common_delivery] == format_date
                        desired_time = item[common_time] == format_time
                        combined_condition = desired_time & desired_delivery

                        desired_delivery_1 = item[common_delivery] == format_date_1
                        desired_time_1 = item[common_time] == format_time_1
                        combined_condition_1 = desired_time_1 & desired_delivery_1

                        if combined_condition.any(): #this is for the rolling case
                            iteration_df_columns.append(item.loc[combined_condition])   #these are before combining vertically
                            #print('Condition 1 Met')


                        elif combined_condition_1.any():
                            iteration_df_columns.append(item.loc[combined_condition_1])   #these are before combining vertically
                            #print('Condition 2 Met')

                        else:
                            print('Neither Condition Works')

                        try:
                            final_rows_df = pd.concat(iteration_df_columns, axis=1)
                            final_df_columns.append(final_rows_df)

                        except:
                            print('CONTINUE_NO_DF_AVAILABLE')


                    dummy_list = []
                    for list in final_df_columns:
                        if len(final_df_columns) != 0:
                            dummy_list.append(1)
                        else:
                            continue
                    if len(dummy_list) == len(reformatted_list):
                        final_horizontal_concat.append(final_df_columns)

                    else:
                        print(f"-----------------------------------------------AT LEAST ONE MISSING DF")



















    organized_list = []

    for category_index in range(0, len(reformatted_list)):
        category_list = []
        for specific_date_time in final_horizontal_concat:
            category_list.append(specific_date_time[category_index])
        organized_list.append(category_list)


    final_list = []
    for organized_categories in organized_list:
        final_categorical_df = pd.concat(organized_categories, axis = 0, ignore_index=True)
        final_list.append(final_categorical_df)

    print(final_list)
    concated_final = pd.concat(final_list, axis=1)


    date_time = concated_final.iloc[:, 0:2]
    date_time[common_time] = date_time[common_time].str.extract(r'^(\d{1,2}:\d{2})')[0]

    concated_final = concated_final.drop([common_delivery, common_time], axis=1)
    concated_final = pd.concat([date_time, concated_final], axis=1)

    print(concated_final)
    concated_final.to_csv(csv_path, index = True)

    with open(pickle_path, 'wb') as file:
        pickle.dump(concated_final, file)

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
    ercot_data_path.mkdir(exist_ok = True)

    pickle_files_path = ercot_data_path.joinpath('Csv_Files')
    pickle_files_path.mkdir(exist_ok=True)

    csv_file_path = os.path.join(ercot_data_path, 'Final_Concated_DF.csv')

    pickle_files_path = ercot_data_path.joinpath('Pickle_Successful')
    pickle_files_path.mkdir(exist_ok = True)

    created_pickle_path = os.path.join(ercot_data_path, 'Final_Pickle_Data.pkl')

    pickle_name_list = [
        '_523_Final_Df.pkl', '_565_Final_Df.pkl', '_732_Final_Df.pkl', '_737_Final_Df.pkl', '_742_Final_Df.pkl'
    ]

    loaded_pickle_list = []
    for i in range (0, len(pickle_name_list)):
        pickle_path = os.path.join(pickle_files_path, pickle_name_list[i])
        with open(pickle_path, 'rb') as file:
            load_pickle_df = pickle.load(file)

        loaded_pickle_list.append(load_pickle_df)

    concat_2D(loaded_pickle_list, csv_file_path, created_pickle_path)