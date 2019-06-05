from seaborn import heatmap
from pandas import DataFrame
import matplotlib.pyplot as plt

# arra = [[376, 41, 5, 17, 61],
#         [10, 453, 2, 3, 32],
#         [20, 19, 357, 18, 86],
#         [8, 9, 8, 456, 19],
#         [16, 15, 8, 3, 458]]

arra = [[372, 42, 27, 6, 53],
[35, 395, 30, 9, 31],
[37, 16, 380, 11, 56],
[75, 21, 63, 323, 18],
[31, 18, 21, 1, 429]]

classes = ["Audi A4", "BMW 3", "VW Golf V", "Mercedes E", "Opel Astra"]

df = DataFrame(arra, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (10,7))
heatmap(df, annot=True)
plt.show()

[[372, 42, 27, 6, 53],
[35, 395, 30, 9, 31],
[37, 16, 380, 11, 56],
[75, 21, 63, 323, 18],
[31, 18, 21, 1, 429]]
