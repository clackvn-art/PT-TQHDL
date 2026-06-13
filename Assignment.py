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

#GD2
#1
# phân tích các mối quan hệ
# thời gian học và completion rate
plt.scatter(pf['avg_session_minutes'], pf['completion_rate'])
plt.xlabel('Thời gian học')
plt.ylabel('Mức độ hoàn thành khóa học')
plt.title('Mối quan hệ giữa thời gian học và mức độ hoàn thành khóa học')
plt.show()

# số video xem và khả năng mua khóa học
plt.scatter(pf['videos_watched'], pf['total_spent_usd'])
plt.xlabel('Số video đã xem')
plt.ylabel('Tổng chi tiêu (USD)')
plt.title('Mối quan hệ giữa số video đã xem và tổng chi tiêu')
plt.show()

# hoạt động AI recommendation vs enrollment
plt.scatter(pf['ai_recommend_enroll'], pf['courses_enrolled'])
plt.xlabel('Hoạt động AI recommendation')
plt.ylabel('Số khóa học đã đăng ký')
plt.title('Mối quan hệ giữa hoạt động AI recommendation và số khóa học đã đăng ký')
plt.show()

#2
#Phân cụm người dùng áp dụng K-Means
# Chuẩn hóa dữ liệu
features = ['avg_session_minutes', 'sessions_per_week', 'completion_rate','quizzes_taken']
X = pf[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
pf['cluster'] = kmeans.fit_predict(X_scaled)
print("Phân cụm người dùng:")
print(pf[['avg_session_minutes', 'completion_rate','quizzes_taken', 'cluster']])

#học nhiều và hoàn thành khóa
high_engagement = pf[(pf['avg_session_minutes'] > 25) & (pf['completion_rate'] > 0.8)]

#học ít
low_activity = pf[pf['avg_session_minutes'] < 10]

#làm nhiều quiz
high_quiz = pf[pf['quizzes_taken'] > 5]

#ít hoạt động
low_engagement = pf[(pf['avg_session_minutes'] < 25) & (pf['quizzes_taken'] < 5)]

#3
#Sử dụng Star Glyphs và Chernoff Faces để biểu diễn hành vi của các nhóm.
#Star Glyphs
def star_glyphs(data, features, title):
    num_features = len(features)
    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    for i in range(len(data)):
        values = data.iloc[i][features].tolist()
        values += values[:1]
        plt.polar(angles, values, marker='o', label=f'User {i}')
    
    plt.title(title)
    plt.show()
star_glyphs(high_engagement, features, high_engagement['cluster'])
star_glyphs(low_activity, features, low_activity['cluster'])
star_glyphs(high_quiz, features, high_quiz['cluster'])
star_glyphs(low_engagement, features, low_engagement['cluster'])

#Chernoff Faces
def chernoff_faces(data, features, title):

    cluster_profile = pf.groupby("cluster").mean(numeric_only=True)

    print(cluster_profile)

    faces_data = pd.DataFrame(
    scaler.fit_transform(cluster_profile),
    columns=cluster_profile.columns,
    index=cluster_profile.index
    )
    data = faces_data
    fig, axes = plt.subplots(1, len(data), figsize=(15, 5))
    for i in range(len(data)):
        values = data.iloc[i][features].tolist()
        face = patches.Circle((0.5, 0.5), 0.4, edgecolor='black', facecolor='none')
        axes[i].add_patch(face)
        axes[i].plot([0.5], [0.5], marker='o', markersize=values[0]*10, color='blue')  # eyes
        axes[i].plot([0.5], [0.3], marker='o', markersize=values[1]*10, color='red')   # nose
        axes[i].plot([0.3, 0.7], [0.2, 0.2], marker='o', markersize=values[2]*10, color='green')  # mouth
        axes[i].set_title(f'User {i}')
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()
chernoff_faces(high_engagement, features, high_engagement['cluster'])
chernoff_faces(low_activity, features, low_activity['cluster'])
chernoff_faces(high_quiz, features, high_quiz['cluster'])
chernoff_faces(low_engagement, features, low_engagement['cluster'])

#4
#Sử dụng Treemap và Dendrogram để thể hiện cấu trúc các nhóm hành vi.
#Tree Map
cluster_counts = pf['cluster'].value_counts()
plt.figure(figsize=(8, 6))

squarify.plot(sizes=cluster_counts.values, label=cluster_counts.index, alpha=0.8)
plt.title('Phân bố số lượng người dùng theo cụm')
plt.axis('off')
plt.show()

#Dendrogram
linkage_matrix = linkage(pf[['avg_session_minutes', 'completion_rate', 'quizzes_taken']], method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('User Index')
plt.ylabel('Distance')
plt.show()
