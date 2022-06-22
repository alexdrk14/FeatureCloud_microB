"""Additional libraries"""
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy

"""Plot libraries"""
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "/mnt/input/"
OUTPUT_PATH = "/mnt/output/"

def data_loading():
    #data_path = "/mnt/input/"
    df_anno = pd.read_csv(INPUT_PATH + 'anno.csv')
    df_exp = pd.read_csv(INPUT_PATH + 'exp.csv')
    df_exp.columns = ['sample'] + df_exp.columns.to_list()[1:]
    data = pd.merge(df_anno, df_exp, on='sample')
    country = data['country'][0]

    """Keep only samples with CRC and healthy target"""
    data = data[data['class'].isin(['CRC', 'healthy'])]

    """Encode categorical features in DataFrame"""
    data, target = encode_categorical(data)

    """Remove unnecessary features"""
    data = feature_filtering(data)
    return data, target, country

"""Function that encodes categorical features in dataframe"""
def encode_categorical(data):
    """Drop first column and keep target separetely"""
    target = [1 if value == 'CRC' else 0 for value in
              data['class']]
    #data['class'] == 'CRC'
    target = [1 if value else 0 for value in target]
    #data['class'].astype('category').cat.code

    data['health_status'] = [1 if value == 'P' else 0 for value in
                             data['health_status']]
    #data['health_status'] == 'P'
    data['gender'] = [1 if value == 'male' else 0 for value in data['gender']]
    return data, target

"""Removes unnecessary features"""
def feature_filtering(data):
    drop_features = ['class', 'sample', 'HQ_clean_read_count', 'gut_mapped_read_count',
                     'gut_mapped_pc', 'oral_mapped_read_count', 'oral_mapped_pc', 'low_read',
                     'low_map', 'excluded', 'excluded_comment', 'sel_Beghini_2021',
                     'study_accession', 'sample_accession', 'secondary_sample_accession',
                     'instrument_platform', 'instrument_model', 'library_layout', 'sample_alias',
                     'mgp_sample_alias', 'westernised', 'country', 'individual_id', 'timepoint', 'body_site',
                     'body_subsite',
                     'host_phenotype', 'host_subphenotype', 'to_exclude']

    data.drop(drop_features, inplace=True, axis=1)
    return data

def feature_NA_manager(X, entropy_threshold):
    # X = X.replace(np.nan, 0)
    drop_out = []
    imputed = []
    for feature in X.columns.to_list():
        if np.isnan(X[feature]).any():
            values = [value for value in X[feature] if value != np.nan]

            if(len(values) == 0) or (len(values) < X.shape[0] / 2) or (entropy([value / len(values) for value in values], base=2) < entropy_threshold):
                drop_out.append(feature)
            else:
                """Value impute"""
                imputed.append(feature)
                if type(values[0]) == np.float64:
                    """In case of float values imbute NaN with avg value"""
                    new_value = sum(values) / len(values)
                else:
                    new_value = Counter(values).most_common(1)[0][0]
                X[feature] = X[feature].replace(np.nan, new_value)
    return X, drop_out, imputed

def plot_confusion(conf_matrix: np.array, country: str, iteration:int):
        """
        Plot the confusion matrix

        :param conf_matrix: Input confusion matrix
        :country: Country of data origin
        :iteration: Number of iteration cicle
        :return:
        """

        fig, ax = plt.subplots(figsize=(15, 12))
        ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', annot_kws={"size": 32})

        ax.set_title(f'Confusion matrix for {country} country \n\n', fontsize=32, fontweight='bold')
        ax.set_xlabel('\nPredicted Values', fontsize=28, fontweight='bold')
        ax.set_ylabel('Actual Values ', fontsize=28, fontweight='bold')

        ax.xaxis.set_ticklabels(['Healthy', 'CRC'], fontsize=28, fontweight='bold')
        ax.yaxis.set_ticklabels(['Healthy', 'CRC'], fontsize=28, fontweight='bold')

        plt.show()
        fig.savefig(OUTPUT_PATH + f"confusion_matrix_{country}_iter{iteration}.png")
        plt.close()


