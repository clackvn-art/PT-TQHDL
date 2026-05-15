import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("moscow.csv")

#Bài 1
print(df.head(10))
print(df.shape)
print(df.dtypes)

AVG = df["rental_price"].mean()
print("Giá trị trung bình:", AVG)

MAX = df['rental_price'].max()
MIN = df["rental_price"].min()
print("Giá trị lớn nhất:", MAX)
print("Giá trị nhỏ nhất:", MIN)

STD = df["rental_price"].std()
print("Độ lệch chuẩn:", STD)

NUM_DF = df.select_dtypes(include=["float64", "int64"])
NUM_DF.hist(figsize=(10, 8))
plt.show()

df["okrug"].value_counts().plot(kind="bar")
plt.show()

#Bài 2
print (df.isnull().sum())
df.fillna(df["secondary_mom_change_pct"].mean(), inplace=True)
df.fillna(df["newbuild_mom_change_pct"].mean(), inplace=True)
print (df.isnull().sum())

print ("số lượng bị trùng:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print ("Sau khi xóa trùng lặp")
print (df.shape)


scaler = StandardScaler()
numeric_cols = df.select_dtypes(include= ["float64", "int64"]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df[numeric_cols].boxplot(figsize=(10, 8))
plt.show()