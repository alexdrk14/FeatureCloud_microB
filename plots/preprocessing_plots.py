"""
    Data preprocessing plots

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-06-21
"""

import os

from bokeh.plotting import figure, output_file, show
import matplotlib.pyplot as plt
import numpy as np


def value_range_hist_bokeh(input_values: list, country_code: str, file_name: str, feature_name: str):
    """
    Plot and store the value range histograms with Bokeh

    :param input_values: List of feature values as input for the histogram plot
    :param country_code: Country code
    :param file_name: File name (anno or exp)
    :param feature_name: Name of feature
    :return:
    """

    output_data_path = os.path.join(os.path.join("data", "output_data", "plots", "value_ranges",
                                                 country_code, file_name))

    tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save," \
            "box_select,poly_select,lasso_select,"
    p_figure = figure(width=1200, height=1200, tools=tools)
    p_figure.title.text = "Range of values: " + feature_name
    p_figure.title.text_font_size = '24pt'

    p_figure.axis.axis_label_text_font_style = 'bold'

    p_figure.xaxis.axis_label = f'Values of {feature_name}'
    p_figure.xaxis.axis_label_text_font_size = "20pt"
    p_figure.xaxis.major_label_text_font_style = 'bold'
    p_figure.xaxis.major_label_text_font_size = "16pt"

    p_figure.yaxis.axis_label = 'Values of histogram'
    p_figure.yaxis.axis_label_text_font_size = "20pt"
    p_figure.yaxis.major_label_text_font_style = 'bold'
    p_figure.yaxis.major_label_text_font_size = "16pt"

    hist, edges = np.histogram(input_values, density=True, bins=50)

    p_figure.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
    output_file(os.path.join(output_data_path, f"Value_Range_{feature_name}.html"))
    # export_png(p_figure, filename=os.path.join(output_data_path, f"Value_Range_{feature_name}.png"))

    show(p_figure)


def value_range_hist_matplotlib(input_values: list, country_code: str, file_name: str, feature_name: str):
    """
    Plot and store the value range histograms with matplotlib

    :param input_values: List of feature values as input for the histogram plot
    :param country_code: Country code
    :param file_name: File name (anno or exp)
    :param feature_name: Name of feature
    :return:
    """

    output_data_path = os.path.join(os.path.join("data", "output_data", "plots", "value_ranges",
                                                 country_code, file_name))

    fig = plt.figure(figsize=(12, 12))

    n, bins, patches = plt.hist(input_values, 50, density=True, facecolor='b', alpha=0.75)

    plt.xlabel("Range of values", fontsize=26, fontweight='bold')
    plt.ylabel(feature_name, fontsize=26, fontweight='bold')
    plt.title("Range of values: " + feature_name, fontsize=32, fontweight='bold')
    # plt.show()
    fig.savefig(os.path.join(output_data_path, f"Value_Range_{feature_name}.png"))
    plt.close()
