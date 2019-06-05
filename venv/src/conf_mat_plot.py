from seaborn import heatmap
from pandas import DataFrame
import matplotlib.pyplot as plt

arra = [[376, 41, 5, 17, 61],
        [10, 453, 2, 3, 32],
        [20, 19, 357, 18, 86],
        [8, 9, 8, 456, 19],
        [16, 15, 8, 3, 458]]

classes = ["Audi A4", "BMW 3", "VW Golf V", "Mercedes E", "Opel Astra"]

df = DataFrame(arra, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (10,7))
heatmap(df, annot=True)
plt.show()