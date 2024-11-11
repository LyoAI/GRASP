import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
matplotlib.rcParams['font.family'] = 'Arial'


def plot_metric_barplot(data: Optional[pd.DataFrame] = None):
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    if not data:
        data = {
            "Singular Value": [i for i in range(10)],
            "Taylor Expansion Value": [4.3, 2, 3.1, 0.8, 0.7, 1.9, 0.2, 0.3, 0.1, 2.5]
        }


    c_list = ['#DF9E9B'] # '#DF9E9B', '#99BADF', '#D8E7CA', '#99CDCE', '#999ACD'
    plt.figure(figsize=(7, 5))
    # plt.grid(axis="y", linestyle='-.')
    ax = sns.barplot(data=data, x='Singular Value', y='Taylor Expansion Value', width=0.7, color="#DF9E9B", saturation=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.yticks(None)
    plt.savefig('./temp.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_metric_barplot()