import pandas as pd
import pickle
import os
import argparse
import pathlib
import re
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def data_single_iteration(path_one_category, category_prefix, every_other_passed, common_time, common_delivery, flag):
    numerical_count = []
    file_order = []
    DST_COLUMN = 'DSTFlag'
    DST_Y = []
    exceptions = []
    for file in os.listdir(path_one_category):
        if file.startswith(category_prefix):
            read_file_df = pd.read_csv(filepath_or_buffer = os.path.join(path_one_category, file))
            numerical_count.append(read_file_df)
            DST_rows = read_file_df[read_file_df[DST_COLUMN] == 'Y']
            file_order.append(file)
            if DST_rows.empty:
                continue
            else:
                print('YES DST AVAILABLE')
                DST_Y.append(DST_rows)
                DST_Y.append(file)
                read_file_df = read_file_df[read_file_df[DST_COLUMN] != 'Y']
    def extract_hour(filename):
        parts = filename.split('.')
        if len(parts) > 4:
            return int(parts[4][:2])
    sorted_filenames = sorted(file_order, key=extract_hour)
    extracted_list = []
    for file in sorted_filenames:
        extracted_append = extract_hour(file)
        extracted_list.append(extracted_append)
    if len(sorted_filenames) != 0:
        success = False
        success_523 = False
        selected_target_index = []
        selected_target_index_523 = []
        for i in range(9):
            target_value = 9 - i
            if target_value in extracted_list:
                target_index = extracted_list.index(target_value)
                read_file_df = pd.read_csv(filepath_or_buffer=os.path.join(path_one_category, sorted_filenames[target_index]))
                success = True
                selected_target_index.append(target_index)
                break
            target_523 = 12 + i
            if target_523 in extracted_list:
                target_index_523 = extracted_list.index(target_523)
                read_file_df = pd.read_csv(filepath_or_buffer=os.path.join(path_one_category, sorted_filenames[target_index_523]))
                success_523 = True
                selected_target_index_523.append(target_index_523)
                break
        if not success and not success_523:
            raise ValueError("No successful file Obtained")
        read_file_df.rename(columns={read_file_df.columns[0]: common_delivery}, inplace=True)
        columns_drop = [col for col in read_file_df.columns if col.startswith('ACTUAL_')]
        read_file_df = read_file_df.drop(columns=columns_drop)
        if flag in read_file_df.columns:
            read_file_df = read_file_df[read_file_df[flag] != 'N']
            read_file_df = read_file_df.reset_index(drop=True)
        read_file_df.rename(columns={read_file_df.columns[1]: common_time}, inplace=True)
        def format_hour(hour):
            return f"{hour}:00"
        read_file_df[common_time] = read_file_df[common_time].astype(str)
        read_file_df[common_time] = read_file_df[common_time].apply(format_hour)
        read_file_df[common_time] = read_file_df[common_time].replace('24:00', '0:00')
        read_file_df[common_time] = read_file_df[common_time].replace('24:00:00', '0:00:00')
        if len(selected_target_index) != 0:
            latest_hour = extract_hour(sorted_filenames[selected_target_index[0]])
            format_time = f'{str(latest_hour)}:00'
            format_date = f'{m_cushion}/{d_cushion}/{y_cushion}'
            latest_hour_1 = extract_hour(sorted_filenames[selected_target_index[0]])
            format_time_1 = f"{str(latest_hour_1)}:00:00"
            format_date_1 = f'{m_cushion}/{d_cushion}/{y_cushion}'
            desired_delivery = read_file_df[common_delivery] == format_date
            desired_time = read_file_df[common_time] == format_time
            combined_condition = desired_time & desired_delivery
            desired_delivery_1 = read_file_df[common_delivery] == format_date_1
            desired_time_1 = read_file_df[common_time] == format_time_1
            combined_condition_1 = desired_time_1 & desired_delivery_1
            lower_bound = 24 - target_index
            uppper_bound = 48 - target_index
            print(format_date)
            if combined_condition.any():  #we may have to consider the case with 2022_06_30 here
                index_desired_row = read_file_df[combined_condition].index[0]
                read_file_df = read_file_df.loc[index_desired_row + lower_bound: index_desired_row + uppper_bound]
            elif combined_condition_1.any():
                index_desired_row_1 = read_file_df[combined_condition_1].index[0]
                read_file_df = read_file_df.loc[index_desired_row_1 + lower_bound: index_desired_row_1 + uppper_bound]
            else:
                exceptions.append('At Least One Condition Does not Work')
                print('At Least One Condition Does not Work')
        else:
            format_date = f'{m_cushion}/{d_cushion}/{y_cushion}'
            print(format_date)
            read_file_df = read_file_df
    if len(numerical_count) == 0:
        every_other_passed = every_other_passed + 1
    else:
        every_other_passed = every_other_passed + 1
    return every_other_passed, read_file_df, exceptions, DST_Y

def data_combine_refine_dowload(master_list, created_csv_path, pickle_path, pickle_name, common_delivery, common_time):
    combined_df_iteration = pd.concat(master_list, axis=0, ignore_index=True)
    combined_df_iteration = combined_df_iteration.drop_duplicates(subset = [common_delivery, common_time], keep='first').reset_index(drop=True)
    date_time_df = combined_df_iteration.iloc[:, 0:2]
    string_columns = combined_df_iteration.select_dtypes(include=['object', 'string']).columns
    individual_df = combined_df_iteration.drop(columns=string_columns)
    combined_df_iteration = pd.concat([date_time_df, individual_df], axis=1)
    print(combined_df_iteration)
    csv_path = created_csv_path
    combined_df_iteration.to_csv(csv_path, index = True)
    file_path = os.path.join(pickle_path, pickle_name)
    with open(file_path, 'wb') as file:
        pickle.dump(combined_df_iteration, file)
    return combined_df_iteration

def concat_2D(list_load, csv_path, pickle_path):
    load_523 = list_load[0]
    load_565 = list_load[1]
    load_732 = list_load[2]
    load_737 = list_load[3]
    load_742 = list_load[4]
    date_1 = '07/01/2022'
    date_3 = '07/03/2022'
    common_time = 'Hour_Ending'
    common_delivery = 'DeliveryDate'
    corrupt_data_1 = load_737[load_737[common_delivery] == (date_1)]
    corrupt_data_3 = load_737[load_737[common_delivery] == (date_3)]
    corrupt_data_1 = corrupt_data_1.iloc[:-1, :]
    corrupt_data_3 = corrupt_data_3.iloc[:-1, :]
    numeric_columns = load_737.select_dtypes(exclude=['object']).columns
    average_rows = []
    for index in range(corrupt_data_1.shape[0]):
        row1 = corrupt_data_1.iloc[index]
        row2 = corrupt_data_3.iloc[index]
        average_row = {}
        for column in corrupt_data_1.columns:
            if column in numeric_columns:
                average_row[column] = (row1[column] + row2[column]) / 2
            else:
                average_row[column] = row1[column]
        average_rows.append(average_row)
    average_df = pd.DataFrame(average_rows)
    average_df['DeliveryDate'] = pd.to_datetime(average_df['DeliveryDate'], format='%m/%d/%Y')
    average_df['DeliveryDate'] = average_df['DeliveryDate'].apply(lambda x: x.replace(year=2022, day=2))
    average_df['DeliveryDate'] = average_df['DeliveryDate'] + pd.DateOffset(hours=1)
    average_df['DeliveryDate'] = average_df['DeliveryDate'].dt.strftime('%m/%d/%Y')
    load_737 = pd.concat([load_737, average_df],axis=0)
    load_737.sort_index(inplace=True)
    load_737.reset_index(inplace=True)
    load_737['DateTime'] = pd.to_datetime(load_737[common_delivery] + ' ' + load_737[common_time])
    load_737 = load_737.sort_values(by='DateTime').reset_index(drop=True)
    load_737 = load_737.drop(columns=['DateTime', 'index'])
    load_737['Hour_Ending'] = load_737['Hour_Ending'].shift(-1)
    load_737['Hour_Ending'].iloc[-1] = '0:00'
    new_list_load = [list_load[0], list_load[1], list_load[2], load_737, list_load[4]]
    columns_to_rename_523 = new_list_load[0].columns[2:]
    new_column_names_523 = {old: old + '_ActualValues' for old in columns_to_rename_523}
    load_523.rename(columns = new_column_names_523, inplace=True)
    columns_to_rename_565 = new_list_load[1].columns[2:]
    new_column_names_565  = {old: old + '_WeatherZone' for old in columns_to_rename_565}
    load_565.rename(columns = new_column_names_565 , inplace=True)
    columns_to_rename_732 = new_list_load[2].columns[2:]
    new_column_names_732  = {old: old + '_WindPowerProduction' for old in columns_to_rename_732}
    load_732.rename(columns = new_column_names_732 , inplace=True)
    columns_to_rename_737 = new_list_load[3].columns[2:]
    new_column_names_737  = {old: old + '_SolarPowerProduction' for old in columns_to_rename_737}
    load_737.rename(columns = new_column_names_737 , inplace=True)
    columns_to_rename_742 = new_list_load[4].columns[2:]
    new_column_names_742 = {old: old + '_WindPowerProductionGeographical' for old in columns_to_rename_742}
    load_742.rename(columns = new_column_names_742, inplace=True)
    def format_hour(hour):
        return f"{hour}:00"
    def check_format(x):
        x = str(x)
        return bool(re.match(r'^\d{1,2}:00$', x))
    reformatted_list = []
    for item in new_list_load:
        boolean_column = item[common_time].apply(check_format)
        if boolean_column.all():
            item[common_time] = item[common_time]
        else:
            item[common_time] = item[common_time].astype(str)
            item[common_time] = item[common_time].apply(format_hour)
        reformatted_list.append(item)
    final_horizontal_concat = []
    start_date = datetime.date(2017, 12, 31)
    end_date = datetime.date(2024, 6, 17) #6/17 is the latest date available
    current_date = start_date
    while current_date <= end_date:
        for hour in range(0, 24):
            y_cushion = f'{current_date.year}'
            m_cushion = f'{current_date.month:02d}'
            d_cushion = f'{current_date.day:02d}'
            format_time = f'{str(hour)}:00'
            format_date = f'{m_cushion}/{d_cushion}/{y_cushion}'
            format_time_1 = f'{str(hour)}:00:00:00'
            format_date_1 = f'{m_cushion}/{d_cushion}/{y_cushion}'
            final_df_columns = []
            for i in range(0, len(reformatted_list)):
                iteration_df_columns = []
                desired_delivery = reformatted_list[i][common_delivery] == format_date
                desired_time = reformatted_list[i][common_time] == format_time
                combined_condition = desired_time & desired_delivery
                desired_delivery_1 = reformatted_list[i][common_delivery] == format_date_1
                desired_time_1 = reformatted_list[i][common_time] == format_time_1
                combined_condition_1 = desired_time_1 & desired_delivery_1
                if combined_condition.any():
                    iteration_df_columns.append(reformatted_list[i].loc[combined_condition])
                    final_rows_df = pd.concat(iteration_df_columns, axis=1)
                    final_df_columns.append(final_rows_df)
                elif combined_condition_1.any():
                    iteration_df_columns.append(reformatted_list[i].loc[combined_condition_1])
                    final_rows_df = pd.concat(iteration_df_columns, axis=1)
                    final_df_columns.append(final_rows_df)
            dummy_list = []
            for list in final_df_columns:
                if len(final_df_columns) != 0:
                    dummy_list.append(1)
                else:
                    continue
            if len(dummy_list) == len(reformatted_list):
                final_horizontal_concat.append(final_df_columns)
        current_date += datetime.timedelta(days=1)
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
    concated_final = pd.concat(final_list, axis=1)
    date_time = concated_final.iloc[:, 0:2]
    date_time[common_time] = date_time[common_time].str.extract(r'^(\d{1,2}:\d{2})')[0]
    concated_final = concated_final.drop([common_delivery, common_time], axis=1)
    concated_final = pd.concat([date_time, concated_final], axis=1)
    print(concated_final)
    concated_final.to_csv(csv_path, index = True)
    with open(pickle_path, 'wb') as file:
        pickle.dump(concated_final, file)

def DST_Filter(pickle_path, csv_file_path_filter):
    common_delivery = 'DeliveryDate'
    common_time = 'Hour_Ending'
    with open(pickle_path, 'rb') as file:
        load_final_df = pickle.load(file)
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
    load_final_df['DateTime'] = pd.to_datetime(load_final_df[common_delivery] + ' ' + load_final_df[common_time])
    load_final_df = load_final_df.sort_values(by='DateTime').reset_index(drop=True)
    load_final_df = load_final_df.drop(columns=['DateTime'])
    load_final_df.to_csv(csv_file_path_filter, index=True)
    return load_final_df

def seperate_df(load_final_df, path_24, pickle_24, pickle_path_updated):
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
    with open(pickle_path_updated, 'wb') as file:
        pickle.dump(load_final_df, file)

def OHE_data(data_path, name_csv):
    df_function = os.path.join(data_path, name_csv)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)
    df = pd.read_csv(df_function)
    common_time = 'Hour_Ending'
    common_delivery = 'DeliveryDate'
    actual_desired_lambda = 'SystemLambda_ActualValues'
    df['DateTime'] = pd.to_datetime(df['DeliveryDate'])
    df['weekday'] = df['DateTime'].dt.weekday
    ohe_weekdays = pd.get_dummies(df['weekday'], prefix='weekday')
    ohe_weekdays = ohe_weekdays.astype(bool).astype(int)
    df['month'] = df['DateTime'].dt.month
    ohe_months = pd.get_dummies(df['month'], prefix='month')
    ohe_months = ohe_months.astype(bool).astype(int)
    df['hour'] = df['Hour_Ending'].str.split(':').str[0].astype(int)
    ohe_hours = pd.get_dummies(df['hour'], prefix='hour')
    ohe_hours = ohe_hours.astype(bool).astype(int)
    train_drop = df.drop(columns=['weekday', 'month', 'hour', common_delivery, common_time, 'DateTime'])
    train_drop = train_drop.iloc[:, 1:]
    before_numbers_train = train_drop.select_dtypes(exclude=['object'])
    before_numbers_train.fillna(0, inplace=True)
    square_df = pd.DataFrame()
    for column in before_numbers_train.columns:
        square_df[f"{column}_squared"] = before_numbers_train[column] ** 2
    for column in before_numbers_train.columns:
        square_df[f"{column}_cube"] = before_numbers_train[column] ** 3
    for column in before_numbers_train.columns:
        square_df[f"{column}_quartic"] = before_numbers_train[column] ** 4
    for column in before_numbers_train.columns:
        square_df[f"{column}_quintic"] = before_numbers_train[column] ** 5
    for column in before_numbers_train.columns:
        square_df[f"{column}_sqrt"] = before_numbers_train[column] ** 0.5
    for column in before_numbers_train.columns:
        square_df[f"{column}_log"] = np.log10(before_numbers_train[column])
    before_numbers_train_concat_data = pd.concat([before_numbers_train, square_df], axis=1)
    before_split_train_manipulated = before_numbers_train_concat_data.apply(lambda x: (x - x.mean()) / (x.std()))
    concat_data = pd.concat([ohe_weekdays, ohe_months, ohe_hours, before_split_train_manipulated], axis=1)
    concat_data = concat_data.select_dtypes(exclude=['object'])
    concat_data.fillna(0, inplace=True)
    csv_file_path = os.path.join(data_path, "../Hot_Encoded_Data_Final_DF.csv")
    pickle_path_updated = os.path.join(data_path, "../Hot_Encode_Final_DF.pkl")
    concat_data.to_csv(csv_file_path, index=True)
    with open(pickle_path_updated, 'wb') as file:
        pickle.dump(concat_data, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data proccesor: Please run the python script from the directory containing this file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--created_directory",
        type=pathlib.Path,
        required=True,
        help="Absolute path to all files that you want to store"
    )
    parser.add_argument(
        "--category_path_list",
        nargs='+',
        type=str,
        required=True,
        help="Please enter the absolute path contatining the csv files in the order below: seperate directories with 1 space:"
             "category_path_523, category_path_565, category_path_732, category_path_737, category_path_742"
    )
    args = parser.parse_args()
    created_directory = args.created_directory
    category_path_list = args.category_path_list
    prefixes_list = ["cdr.00013113.0000000000000000.",
                     "cdr.00014837.0000000000000000.",
                     "cdr.00013028.0000000000000000.",
                     "cdr.00013483.0000000000000000.",
                     "cdr.00014787.0000000000000000."
                     ]
    pickle_name_list = [
        '_523_Final_Df.pkl', '_565_Final_Df.pkl', '_732_Final_Df.pkl', '_737_Final_Df.pkl', '_742_Final_Df.pkl'
    ]
    csv_list_name = [
        '_523_Sheet.csv', '_565_Sheet.csv', '_732_Sheet.csv', '_737_Sheet.csv', '_742_Sheet.csv'
    ]
    ercot_data_path = created_directory.joinpath('ERCOT_Data_Final')
    ercot_data_path.mkdir(exist_ok=True)
    csv_files_path = ercot_data_path.joinpath('Csv_Files_Individual_Categories')
    csv_files_path.mkdir(exist_ok=True)
    pickle_files_path = ercot_data_path.joinpath('Pickle_Successful_Individual_Categories')
    pickle_files_path.mkdir(exist_ok=True)
    OHE_folder = ercot_data_path.joinpath('Combined_Data_Before_OHE')
    OHE_folder.mkdir(exist_ok=True)
    csv_file_path = os.path.join(OHE_folder, 'Concated_DF.csv')
    csv_file_path_filter = os.path.join(OHE_folder, 'Concated_DF_DST_Filtered.csv')
    created_pickle_path = os.path.join(OHE_folder, 'Pickle_Concat.pkl')
    pickle_path_updated = os.path.join(OHE_folder, 'Pickle_Concat_DST_Filtered.pkl')
    path_24 = ercot_data_path.joinpath('24_Hour_Data_OHE')
    path_24.mkdir(exist_ok=True)
    pickle_24 = ercot_data_path.joinpath('24_Hour_Pickle_OHE')
    pickle_24.mkdir(exist_ok=True)
    common_time = 'Hour_Ending'
    common_delivery = 'DeliveryDate'
    flag = 'InUseFlag'
    total_exceptions = 0
    category_combined_list = []
    for j in range(0, len(prefixes_list)):
        master_list = []
        exception_iteration = []
        created_csv_path = csv_files_path.joinpath(csv_list_name[j])
        total_y = []
        start_date = datetime.date(2017, 12, 31)
        end_date = datetime.date(2024, 6, 17)  # 6/17 is the latest date available
        current_date = start_date
        while current_date <= end_date:
            y_cushion = f'{current_date.year}'
            m_cushion = f'{current_date.month:02d}'
            d_cushion = f'{current_date.day:02d}'
            date_string = f"{y_cushion}{m_cushion}{d_cushion}"
            formatted_prefix = f"{prefixes_list[j]}{date_string}"
            every_other, single_iteration_data, length_exceptions, DST_Y = data_single_iteration(category_path_list[j], formatted_prefix, current_date.day, common_time, common_delivery, flag)
            for i in range(0,len(length_exceptions)):
                exception_iteration.append(length_exceptions[i])
            total_exceptions = total_exceptions + len(length_exceptions)
            master_list.append(single_iteration_data)
            current_date += datetime.timedelta(days=1)
        combined_df_iteration = data_combine_refine_dowload(master_list, created_csv_path, pickle_files_path, pickle_name_list[j], common_delivery, common_time)
        category_combined_list.append(combined_df_iteration)
    concat_2D(category_combined_list, csv_file_path, created_pickle_path)
    DST_concat = DST_Filter(created_pickle_path, csv_file_path_filter)
    OHE_data(OHE_folder, "Concated_DF_DST_Filtered.csv")
    seperate_df(DST_concat, path_24, pickle_24, pickle_path_updated)
    print(f"Total exceptions throughout entire proccess: {total_exceptions}")
