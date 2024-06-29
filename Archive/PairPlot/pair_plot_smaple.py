import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setup data
np.random.seed(0)
data = np.random.randn(100, 5)
categories = np.random.choice(['Group 1', 'Group 2', 'Group 3'], size=100)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
df['Category'] = categories

# Compute the correlation matrix
corr = df.select_dtypes(include=[np.number]).corr()

# Prepare the figure and axes
fig, axes = plt.subplots(5, 5, figsize=(20, 20))

# Define the color map and palette
cmap = sns.diverging_palette(220, 20, as_cmap=True)
palette = sns.color_palette("Set2", n_colors=3)

for i in range(5):
    for j in range(5):
        ax = axes[i][j]
        if i != j:
            if i < j:
                # Display correlation coefficients in a heatmap
                sns.heatmap(pd.DataFrame([corr.iloc[i, j]]), annot=True, fmt=".2f", cmap=cmap, ax=ax, cbar=False, vmin=-1, vmax=1, annot_kws={'size': 12, 'weight': 'bold'})
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                # Scatter plot for other cells
                sns.scatterplot(data=df, x=df.columns[j], y=df.columns[i], hue="Category", palette=palette, ax=ax, legend=False)
        else:
            # KDE plots on the diagonal
            sns.kdeplot(data=df, x=df.columns[i], hue="Category", fill=True, palette=palette, ax=ax)
            ax.get_legend().remove()

# Adjusting layout
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Add a colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # Smaller and more refined placement
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

# Global legend for the categories
handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.95), title="Category")

# Save and show the plot
plt.savefig('final_adjusted_plot.png')
plt.show()
