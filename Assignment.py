import pandas as pd
import matplotlib.pyplot as plt

#GD1
#1
#nguồn dữ liệu
pf = pd.read_csv("learnx.csv")

#số bản ghi và thuộc tính
print(pf.shape)
print("Thuộc tính:")
print(pf.dtypes)

#2
#missing value
print("Số lượng giá trị thiếu:")
print(pf.isnull().sum())

#dự liệu bất thường
print("Dữ liệu bất thường:")
print(pf.describe())

#dữ liệu trùng lặp
print("Số lượng dữ liệu trùng lặp:")
print(pf.duplicated().sum())

#3
#biểu đồ phân phối thời gian học
plt.hist(pf['avg_session_minutes'], bins=20)
plt.xlabel('Thời gian học')
plt.ylabel('Số lượng')
plt.title('Phân phối thời gian học')
plt.show()

#biểu đồ số lần truy cập mỗi tuần
plt.hist(pf['sessions_per_week'], bins=20)
plt.xlabel('Số lần truy cập mỗi tuần')
plt.ylabel('Số lượng')
plt.title('Phân phối số lần truy cập mỗi tuần')
plt.show()

#biểu đồ mức độ hoàn thành khóa học
plt.hist(pf['completion_rate'], bins=20)
plt.xlabel('Mức độ hoàn thành khóa học')
plt.ylabel('Số lượng')
plt.title('Phân phối mức độ hoàn thành khóa học')
plt.show()

#4
#Phát hiện các hành vi bất thường
#người dùng học cực kỳ nhiều
outliers = pf[pf['avg_session_minutes'] > 10]
print("Người dùng học cực kỳ nhiều:")
print(outliers)

#người dùng đăng ký nhiều khóa nhưng không học
outliers = pf[(pf['courses_enrolled'] > 3) & (pf['avg_session_minutes'] < 10)]
print("Người dùng đăng ký nhiều khóa nhưng không học:")
print(outliers)

#người dùng chi tiêu bất thường
outliers = pf[pf['total_spent_usd'] > 120]
print("Người dùng chi tiêu bất thường:")
print(outliers)
