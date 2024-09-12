import pandas as pd
import pickle
import os
import argparse
import pathlib
import re

def data_single_iteration(path_one_category, category_prefix, every_other_passed, common_time, common_delivery, flag):
    path_list = path_one_category

    common_time = common_time
    common_delivery = common_delivery
    flag = flag

    prefix = category_prefix
    numerical_count = []
    file_order = []
    DST_COLUMN = 'DSTFlag'
    DST_Y = []

    every_other = every_other_passed

    exceptions = []  # ------------------------------------------------------------------------------------------
    for file in os.listdir(path_list):
        if file.startswith(prefix):
            try:
                read_file_df = pd.read_csv(filepath_or_buffer = os.path.join(path_list, file))  # ------------------------------------------------------------------------------------------
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

            except:
                exceptions.append(file)
                continue

    def extract_hour(filename):
        parts = filename.split('.')
        if len(parts) > 4:
            return int(parts[4][:2])
        return 'error'

    sorted_filenames = sorted(file_order, key=extract_hour)
    extracted_list = []

    for file in sorted_filenames:
        extracted_append = extract_hour(file)
        extracted_list.append(extracted_append)


    if len(sorted_filenames) != 0:
        dummy_list_ = []
        for i in range(0, 9):
            try:
                if len(dummy_list_) == 0:
                    target_index = extracted_list.index(9 - i)  # if the file is
                    read_file_df = pd.read_csv(filepath_or_buffer = os.path.join(path_list,sorted_filenames[target_index]))
                    dummy_list_.append(1)
                else:
                    raise ValueError("Best File Name Already Obtained")

            except:
                dummy_list_.append(1)
                exceptions.append(i)
                continue

        try:
            read_file_df.rename(columns={read_file_df.columns[0]: common_delivery}, inplace=True)
            columns_drop = [col for col in read_file_df.columns if col.startswith('ACTUAL_')]
            read_file_df = read_file_df.drop(columns=columns_drop)

            try:
                read_file_df = read_file_df[read_file_df[flag] != 'N']  # for category number 565
                read_file_df = read_file_df.reset_index(drop=True)
                print('Filtered InFLAG for 565')
            except:
                exceptions.append('NO Flags Needed')

            read_file_df.rename(columns={read_file_df.columns[1]: common_time}, inplace=True)

            def format_hour(hour):
                return f"{hour}:00"

            read_file_df[common_time] = read_file_df[common_time].astype(str)
            read_file_df[common_time] = read_file_df[common_time].apply(format_hour)
            read_file_df[common_time] = read_file_df[common_time].replace('24:00', '0:00')
            read_file_df[common_time] = read_file_df[common_time].replace('24:00:00', '0:00:00')


            latest_hour = extract_hour(sorted_filenames[target_index])
            format_time = f'{str(latest_hour)}:00'
            format_date = f'{m_cushion}/{d_cushion}/20{y_cushion}'

            latest_hour_1 = extract_hour(sorted_filenames[target_index])
            format_time_1 = f"{str(latest_hour_1)}:00:00"
            format_date_1 = f'{m_cushion}/{d_cushion}/20{y_cushion}'

            desired_delivery = read_file_df[common_delivery] == format_date
            desired_time = read_file_df[common_time] == format_time
            combined_condition = desired_time & desired_delivery

            desired_delivery_1 = read_file_df[common_delivery] == format_date_1
            desired_time_1 = read_file_df[common_time] == format_time_1
            combined_condition_1 = desired_time_1 & desired_delivery_1


            print(format_date)

            lower_bound = 24 - target_index
            uppper_bound = 48 - target_index

            if combined_condition.any():
                index_desired_row = read_file_df[combined_condition].index[0]


                read_file_df = read_file_df.loc[index_desired_row + lower_bound: index_desired_row + uppper_bound]  # this way, we include the best prediction at 9 clock TMW, which is still going to be available to us-

            elif combined_condition_1.any():
                index_desired_row_1 = read_file_df[combined_condition_1].index[0]
                read_file_df = read_file_df.loc[index_desired_row_1 + lower_bound: index_desired_row_1 + uppper_bound]  # this is for the case where we have rolling times and
            else:
                exceptions.append('At Least One Condition Does not Work')
                print('At Least One Condition Does not Work')

        except:
            exceptions.append("Error_outer_loop")
            print("Error_outer_loop")

    if len(numerical_count) == 0:
        every_other = every_other + 1
    else:
        #________________________________________________________________________this is where we can do every other return
        #print('Working')
        every_other = every_other + 1



    try:
        return every_other, read_file_df, exceptions, DST_Y

    except:
        print('NO Df Available')

        exceptions.append('NO Df Available')
        return every_other, exceptions, DST_Y


def data_combine_refine_dowload(master_list, created_csv_path, pickle_path, pickle_name, csv_individual_sheet_name, common_delivery, common_time):
    combined_df_iteration = pd.concat(master_list, axis=0, ignore_index=True)
    combined_df_iteration = combined_df_iteration.drop_duplicates(subset = [common_delivery, common_time], keep='first').reset_index(drop=True)  # when two are same, we keep the FIRST APPEARENCE

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




def seperate_df(pickle_path, path_24, pickle_24, csv_file_path, pickle_path_updated):
    common_delivery = 'DeliveryDate'
    common_time = 'Hour_Ending'

    with open(pickle_path, 'rb') as file:
        load_final_df = pickle.load(file)

    print(load_final_df)
    DST_Dates = [['03/11/2018', '11/04/2018'], ['03/10/2019', '11/03/2019'], ['03/08/2020', '11/01/2020'],
                 ['03/14/2021', '11/07/2021'], ['03/13/2022', '11/06/2022'], ['03/12/2023', '11/05/2023'],
                 ['03/10/2024', '11/03/2024']]
    # double 2 o clock, skip 3 o clock, '03/10/2024', '11/03/2024' 2023 March 12-Nov 5, 2022: mar 13, Nov 6,  2021: mar 14, nov 7, 2020: mar 8, nov 1 2019: mar 10, nov 3, 2018: mar 11, nov 4

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
