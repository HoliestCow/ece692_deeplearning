
import pandas as pd


data = [
    [1, 2, 'lol'],
    [3, 4, 'rofl'],
    [5, 6, 'lol']
]

result = pd.DataFrame.from_records(data,
                                   columns=['number1', 'number2', 'label'],
                                   index=['label'])