import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
Medical Data Visualizer

Implements the FreeCodeCamp project tasks:
1) Load data and add an 'overweight' column based on BMI > 25.
2) Normalize 'cholesterol' and 'gluc' so that 0 is good (value == 1) and 1 is bad (value > 1).
3) Draw a categorical plot comparing counts per variable split by cardio.
4) Clean data and draw a correlation heatmap.
"""

# 1 - Import data
df = pd.read_csv('medical_examination.csv')

# 2 - Add 'overweight' column (BMI > 25 -> 1 else 0)
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3 - Normalize cholesterol and gluc (0 is good, 1 is bad)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4
def draw_cat_plot():
    # 5 - Prepare data for categorical plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # 6 - Group and reformat the data to show the counts
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()
    
    # 7 - Create the categorical plot
    g = sns.catplot(
        data=df_cat,
        x='variable', y='total', hue='value', col='cardio', kind='bar'
    )

    # 8 - Get the figure and set labels
    g.set_axis_labels('variable', 'total')
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11 - Clean the data
    df_heat = df.copy()

    # Filter incorrect blood pressure data
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]

    # Filter height and weight by 2.5th and 97.5th percentiles
    h_low, h_high = df_heat['height'].quantile([0.025, 0.975])
    w_low, w_high = df_heat['weight'].quantile([0.025, 0.975])
    df_heat = df_heat[(df_heat['height'] >= h_low) & (df_heat['height'] <= h_high)]
    df_heat = df_heat[(df_heat['weight'] >= w_low) & (df_heat['weight'] <= w_high)]

    # 12 - Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # 13 - Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 - Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
