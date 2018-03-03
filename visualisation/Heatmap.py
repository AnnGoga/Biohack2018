import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import sys

# get and create data
print('Hello username:',sys.argv[])
data = pd.read_csv(sys.argv[1])

# print and save heatmap
sb.heatmap(df, annot=True)
plt.savefig('Heatmap.png', dpi=100)
plt.show()