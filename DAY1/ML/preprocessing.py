import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("DAY1/Iris.csv")
df.head()

features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

df.isnull().sum()

df.set_index('Id', inplace=True)

plt.hist(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],bins=8)
plt.xlabel("Measurment in cm")
plt.ylabel("Frequency")
plt.title("Data Distribution of Iris Dataset")
plt.legend(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.show()

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df)
plt.title("Scatter Plot #1")
plt.show()

sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df)
plt.title("Scatter Plot #2")
plt.show()

ndf=df.select_dtypes(include='number')
corr=ndf.corr()

plt.figure(figsize=(11,10))
sns.heatmap(corr,annot=True, annot_kws={"size": 14}, cmap='coolwarm', fmt='.2f')
plt.show()