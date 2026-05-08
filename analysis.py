import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

class analysis:
    def graf(models,y_test,best_pred,best_name):

        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, best_pred, alpha=0.5)

        min_val = min(y_test.min(), best_pred.min())
        max_val = max(y_test.max(), best_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (идеал)')

        z = np.polyfit(y_test, best_pred, 1)
        plt.plot([min_val, max_val], np.polyval(z, [min_val, max_val]), 'g-',
                 label=f"тренд: y={z[0]:.2f}x+{z[1]:.2f}")

        plt.xlabel('Реальные')
        plt.ylabel('Предсказанные')
        plt.title(f'{best_name}\nR² = {r2_score(y_test, best_pred):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def hist(data):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data)
        plt.title("Гистограмма целевой переменной")
        plt.show()

    def plot_relationship(df, feature, target='medv'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot с регрессией
        sns.regplot(data=df, x=feature, y=target,
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[0])
        axes[0].set_title(f'{feature} vs {target}')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel(target)

        # Hexbin plot (лучше для больших данных)
        hexbin = axes[1].hexbin(df[feature], df[target], gridsize=20, cmap='YlOrRd')
        plt.colorbar(hexbin, ax=axes[1], label='Count')
        axes[1].set_title(f'{feature} vs {target} (плотность)')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel(target)

        plt.tight_layout()
        plt.show()