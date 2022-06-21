"""
    PyMOO (PyMOO) plots

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-06-21
"""


import matplotlib.pyplot as plt


def pymoo_plots(X, F, problem):
    """
    Pymoo plots

    :param X:
    :param F:
    :param problem:
    :return:
    """

    xl, xu = problem.bounds()

    # [1.] Plot design space -------------------------------------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")
    plt.show()
    plt.close()

    # [2.] Plot the objective space ------------------------------------------------------------------------------------

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.show()
    plt.close()

