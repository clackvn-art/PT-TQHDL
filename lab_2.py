import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

df = pd.read_csv("moscow.csv")

def bai_1():
    # chọn 3 tính chất để vẽ histogram
    plt.hist(df['secondary_price_per_sqm'], bins=20)
    plt.hist(df['newbuild_price_per_sqm'], bins=20)
    plt.hist(df['rental_price'], bins=20)
    plt.show()

    #density plot
    sns.kdeplot(df['secondary_price_per_sqm'], label='secondary_price_per_sqm', fill=True)
    sns.kdeplot(df['newbuild_price_per_sqm'], label='newbuild_price_per_sqm', fill=True)
    sns.kdeplot(df['rental_price'], label='rental_price', fill=True)
    plt.show()

    sns.boxplot(data=df['secondary_price_per_sqm'])
    sns.boxplot(data=df['newbuild_price_per_sqm'])
    sns.boxplot(data=df['rental_price'])
    plt.show()
# bai_1()

def bai_2():
    ticket = "AAPL"
    stock = yf.download(ticket, start="2022-01-01", end="2024-01-01")
    stock.head()

    #line chart giá đóng cửa
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Stock Price Over Time")
    plt.plot(stock.index, stock['Close'])
    plt.show()

    #moving average
    stock["MA50"] = stock["Close"].rolling(window=50).mean()
    stock["MA200"] = stock["Close"].rolling(window=200).mean()
    plt.plot(stock.index, stock['Close'], label='Close Price')
    plt.plot(stock.index, stock['MA50'], label='50-day MA')
    plt.plot(stock.index, stock['MA200'], label='200-day MA')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price with Moving Averages")
    plt.legend()
    plt.show()
# bai_2()

def bai_3():
    sns.boxplot(data=df['secondary_price_per_sqm'])
    sns.boxplot(data=df['newbuild_price_per_sqm'])
    sns.boxplot(data=df['rental_price'])
    sns.boxplot(data=df['n_listings_secondary'])
    sns.boxplot(data=df['n_listings_rental'])
    plt.show()
# bai_3()

def bai_4():
    #(correlation matrix)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
bai_4()