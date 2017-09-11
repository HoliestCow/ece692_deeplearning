
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib._color_data as mcd


data = [
    [1, 2, 1, 0.5],
    [3, 4, 1, 0.5],
    [5, 6, 1, 0.5]
]

result = pd.DataFrame.from_records(data,
                                   columns=['number1', 'number2', 'label', 'sublabel'],
                                   index=['label', 'sublabel'])

fig = plt.figure()
ax = plt.gca()
print(result.loc[1, 0.5])
homie = result.loc[(slice(None), 0.5), :]
print(mcd.XKCD_COLORS.keys())
homie.plot(x='number1', y='number2', ax=ax, c=mcd.XKCD_COLORS["xkcd:blue"].upper())
fig.savefig('test.png')

