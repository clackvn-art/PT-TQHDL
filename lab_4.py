import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("moscow.csv")

#bai 1
def heatmap():
    corr_matrix = df.corr(numeric_only=True)
    corr_matrix

    #sns
    plt.figure(figsize=(8,6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5
        )
    plt.title("Correlation Heatmap")
    plt.show()
heatmap()

#bai 2
def Pixel_based_Visualization():
    #Chuẩn bị dataset có nhiều bản ghi
    values = df['secondary_price_per_sqm'].values

    #Biểu diễn mỗi bản ghi bằng một pixel
    size = int(np.ceil(np.sqrt(len(values))))
    pixel_matrix = np.zeros(size*size)
    pixel_matrix[:len(values)] = values
    pixel_matrix = pixel_matrix.reshape(size,size)

    #Sử dụng màu sắc để biểu diễn giá trị dữ liệu
    plt.figure(figsize=(6,6))
    plt.imshow(pixel_matrix, cmap="viridis")
    plt.colorbar()
    plt.title("Pixel-based Visualization")
    plt.show()
Pixel_based_Visualization()

#bai 3
def Star_Glyph():
    #Chọn dataset có nhiều thuộc tính
    attributes = ['secondary_price_per_sqm', 'secondary_mom_change_pct']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[attributes])

    #Vẽ Star Glyph cho từng bản ghi dữ liệu
    def star_plot(values, label):
        num_vars = len(values)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        values = np.concatenate((values,[values[0]]))
        angles = np.concatenate((angles,[angles[0]]))
        fig = plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.3)
        ax.set_title(label)
        plt.show()

    for i in range(3):
        star_plot(scaled_data[i], f"Sample {i}")
Star_Glyph()
    
#bai 4
def Chernoff_Faces():
    def draw_face(ax, data):
        #Chọn dataset có nhiều thuộc tính
        data = [0.8, 0.6, 0.9, 0.4, 0.7]
        ax = plt.subplot(111)

        face_size = 0.5 + data[0] * 0.5
        eye_size = 0.05 + data[1] * 0.05
        mouth_curve = data[2] - 0.5
        nose_size = 0.05 + data[3] * 0.05

        # Vẽ khuôn mặt
        face = plt.Circle((0.5,0.5), face_size*0.4, fill=False, linewidth=2)
        ax.add_patch(face)

        # Vẽ mắt
        left_eye = plt.Circle((0.35,0.6), eye_size, color="black")
        right_eye = plt.Circle((0.65,0.6), eye_size, color="black")
        ax.add_patch(left_eye)
        ax.add_patch(right_eye)

        # Vẽ mũi
        nose = plt.Circle((0.5,0.5), nose_size, color="black")
        ax.add_patch(nose)

        # Vẽ miệng
        x = np.linspace(0.35,0.65,100)
        y = 0.35 + mouth_curve*(x-0.5)**2 * -4
        ax.plot(x,y,linewidth=2)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.axis("off")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        draw_face(ax, df.iloc[i])
        ax.set_title(f"Data {i+1}")

    plt.suptitle("Chernoff Faces Visualization")
    plt.show()
Chernoff_Faces()
