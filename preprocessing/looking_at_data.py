"""
    Looking at data

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-06-21
"""

import glob
import os

import pandas as pd

from plots.preprocessing_plots import value_range_hist_bokeh, value_range_hist_matplotlib


def look_at_data():
    """
    Preprocess the dataframe containing the data that will be used for prediction.

    Plots can help us detect outliers - and further quantify them.
    Correlations (linear and non-linear) between the columns can be computed to make our model smaller.

    :return:
    """

    data_input_folder = os.path.join("data", "microbiome_challenge")

    # [1.] Import the data ---------------------------------------------------------------------------------------------

    all_data_dict = {}

    columns_containing_nan_all_anno = set()
    columns_containing_nan_all_exp = set()

    data_country_subfolders = [x[0] for x in os.walk(data_input_folder)]
    for data_country_folder in data_country_subfolders:

        for country_folder in os.listdir(data_country_folder):

            country_absolute_path = os.path.join(data_country_folder, country_folder)

            if os.path.isdir(country_absolute_path):

                print(country_folder)

                for file_name in os.listdir(country_absolute_path):

                    file_name_absolute_path = os.path.join(country_absolute_path, file_name)
                    if os.path.isfile(file_name_absolute_path):
                        if file_name == "anno.csv":

                            anno_df = pd.read_csv(file_name_absolute_path)
                            anno_df_column_names = list(anno_df.columns)

                            # print(anno_df.head(5))
                            anno_df = anno_df.iloc[:, 1:]
                            # print(anno_df.head(5))

                            print(f"Shape of anno_df: {anno_df.values.shape}")
                            anno_df_nan_values = anno_df.isnull().sum().sum()
                            print(f"Number of NaN values in anno df: {anno_df_nan_values}")

                            columns_containing_nan = anno_df.columns[anno_df.isna().any()].tolist()
                            columns_containing_nan_all_anno.update(columns_containing_nan)
                            for column_containing_nan in columns_containing_nan:
                                nan_nr = anno_df[column_containing_nan].isna().sum()
                                print(f"NaNs in {column_containing_nan}: {nan_nr}")

                            # for column_name in anno_df and column_name not in columns_containing_nan_all_anno:
                            #
                            #    print(country_folder, column_name, "anno")

                            #    column_values_list = list(anno_df[column_name].values)
                            #    value_range_hist_bokeh(column_values_list, country_folder, "anno", column_name)
                            #    value_range_hist_matplotlib(column_values_list, country_folder, "anno", column_name)

                        elif file_name == "exp.csv":

                            exp_df = pd.read_csv(file_name_absolute_path)
                            exp_df_column_names = list(exp_df.columns)

                            # print(exp_df.head(5))
                            exp_df = exp_df.iloc[:, 1:]
                            # print(exp_df.head(5))

                            print(f"Shape of exp_df: {exp_df.values.shape}")
                            exp_df_nan_values = exp_df.isnull().sum().sum()
                            print(f"Number of NaN values in exp df: {exp_df_nan_values}")

                            columns_containing_nan = exp_df.columns[exp_df.isna().any()].tolist()
                            columns_containing_nan_all_exp.update(columns_containing_nan)
                            for column_containing_nan in columns_containing_nan:
                                nan_nr = exp_df[column_containing_nan].isna().sum()
                                print(f"NaNs in {column_containing_nan}: {nan_nr}")

                            # for column_name in exp_df:

                            #    print(country_folder, column_name, "exp")

                            #    column_values_list = list(exp_df[column_name].values)
                            #    value_range_hist_bokeh(column_values_list, country_folder, "exp", column_name)
                            #    value_range_hist_matplotlib(column_values_list, country_folder, "exp", column_name)

                print("-----------------------------------------------------------------------------------------------")

    print(f"Columns containing NaN in anno: {columns_containing_nan_all_anno}")
    print(f"Columns containing NaN in exp: {columns_containing_nan_all_exp}")
