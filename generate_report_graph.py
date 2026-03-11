import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Results from our rectified run
data = {
    'Metric': ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@1', 'Recall@3', 'Recall@5'],
    'Value': [22.22, 44.44, 77.78, 88.89, 100.0, 100.0],
    'Model': ['Self-Attention (1-Head)', 'Self-Attention (1-Head)', 'Self-Attention (1-Head)',
              'Multi-Head (8-Heads)', 'Multi-Head (8-Heads)', 'Multi-Head (8-Heads)']
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
palette = {'Self-Attention (1-Head)': '#3498db', 'Multi-Head (8-Heads)': '#e74c3c'}

ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette=palette)

# Styling
plt.title('Retrieval Performance Comparison: Self-Attention vs Multi-Head', fontsize=16, fontweight='bold')
plt.ylabel('Recall Percentage (%)', fontsize=12)
plt.xlabel('Metric (Top-K)', fontsize=12)
plt.ylim(0, 115)

# Add values on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f') + '%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=11, fontweight='bold')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/final_comparison_bar_graph.png', dpi=300)
print("Bar graph generated at results/final_comparison_bar_graph.png")
