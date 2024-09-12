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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Final_Data_Proccessor INPUTS: Please run the python script from the parent directory',
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

    prefixes_list = ["cdr.00013113.0000000000000000.20{y_cushion}{m_cushion}{d_cushion}",
                     "cdr.00014837.0000000000000000.20{y_cushion}{m_cushion}{d_cushion}",
                     "cdr.00013028.0000000000000000.20{y_cushion}{m_cushion}{d_cushion}",
                     "cdr.00013483.0000000000000000.20{y_cushion}{m_cushion}{d_cushion}",
                     "cdr.00014787.0000000000000000.20{y_cushion}{m_cushion}{d_cushion}"
                     ]

    pickle_name_list = [
        '_523_Final_Df.pkl', '_565_Final_Df.pkl', '_732_Final_Df.pkl', '_737_Final_Df.pkl', '_742_Final_Df.pkl'
    ]

    csv_list_name = [
        '_523_Sheet.csv', '_565_Sheet.csv', '_732_Sheet.csv', '_737_Sheet.csv', '_742_Sheet.csv'
    ]
    csv_file_name = 'All_Proccessed_Data.csv'



    ercot_data_path = created_directory.joinpath('ERCOT_Data')
    ercot_data_path.mkdir(exist_ok=True)

    csv_files_path = ercot_data_path.joinpath('Csv_Files')
    csv_files_path.mkdir(exist_ok=True)


    pickle_files_path = ercot_data_path.joinpath('Pickle_Successful')
    pickle_files_path.mkdir(exist_ok=True)

    common_time = 'Hour_Ending'
    common_delivery = 'DeliveryDate'

    # actual = ['ACTUAL_SYSTEM_WIDE', 'ACTUAL_LZ_NORTH', 'ACTUAL_LZ_WEST', 'ACTUAL_LZ_SOUTH_HOUSTON', 'ACTUAL_SYSTEM_WIDE']
    flag = 'InUseFlag'
    total_exceptions = 0
    total_exception_list = []
    actual_exception_list = []

    DST_Y_Total = []

    DST_Dates = [['03/11/2018', '11/04/2018'], ['03/10/2019', '11/03/2019'], ['03/08/2020', '11/01/2020'], ['03/14/2021', '11/07/2021'], ['03/13/2022', '11/06/2022'], ['03/12/2023', '11/05/2023'], ['03/10/2024', '11/03/2024']]
    # double 2 o clock, skip 3 o clock, '03/10/2024', '11/03/2024' 2023 March 12-Nov 5, 2022: mar 13, Nov 6,  2021: mar 14, nov 7, 2020: mar 8, nov 1 2019: mar 10, nov 3, 2018: mar 11, nov 4






    for j in range(0, len(prefixes_list)):
        master_list = []
        exception_iteration = []
        created_csv_path = csv_files_path.joinpath(csv_list_name[j])  # Use a variable like excel_file_name
        total_y = []
        for y in range(18, 25):  # 18,25
            for m in range(1, 13):  # 1,13
                every_other_updated = 1
                while every_other_updated in range(1, 32):  # 1,32
                    y_cushion = f'{y:02}'
                    m_cushion = f'{m:02}'
                    d_cushion = f'{every_other_updated:02}'

                    formatted_prefix = prefixes_list[j].format(y_cushion=y_cushion, m_cushion=m_cushion, d_cushion=d_cushion)

                    try:
                        every_other, single_iteration_data, length_exceptions, DST_Y = data_single_iteration(category_path_list[j], formatted_prefix, every_other_updated, common_time, common_delivery, flag)

                    except:
                         every_other, length_exceptions, DST_Y = data_single_iteration(category_path_list[j], formatted_prefix, every_other_updated, common_time, common_delivery, flag)
                    every_other_updated= every_other

                    for i in range(0,len(length_exceptions)):
                        exception_iteration.append(length_exceptions[i])

                    total_exceptions = total_exceptions + len(length_exceptions)

                    try:
                        master_list.append(single_iteration_data)
                    except:
                        print('Dataset Not Available')

        data_combine_refine_dowload(master_list, created_csv_path, pickle_files_path, pickle_name_list[j], csv_file_name, common_delivery, common_time)


