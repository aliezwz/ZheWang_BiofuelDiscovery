import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import  numpy as np

data = pd.read_csv("Orignal Dataset.csv")

selected_feature = ['MinPartialCharge', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'Chi2v','VSA_EState8', "Type"]

df = data[selected_feature]

print(df.head())

corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, fmt= ".2f")
plt.show()

fig,axes = plt.subplots(5,5, figsize = (20,20))

cmap = sns.diverging_palette(220, 20, as_cmap=True)
palette = sns.color_palette("Set2", n_colors=2)

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
                sns.scatterplot(data=df, x=df.columns[j], y=df.columns[i], hue="Type", palette=palette, ax=ax, legend=False)
        else:
            # KDE plots on the diagonal
            sns.kdeplot(data=df, x=df.columns[i], hue="Type", fill=True, palette=palette, ax=ax)
            ax.get_legend().remove()


plt.subplots_adjust(hspace=0.5, wspace=0.5)

cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # Smaller and more refined placement
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)

handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.88, 0.95), title="Types")

plt.savefig('final_adjusted_plot.png')
plt.show()