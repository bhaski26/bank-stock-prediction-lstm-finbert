import yfinance as yf
import pandas as pd

# List of bank stocks
#banks = ["SBIN.NS","HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS","KOTAKBANK.NS"]

# Download data
#data = yf.download(
   # banks,
    #start="2015-01-01",
    #end="2024-01-01"
#)

sbi = yf.download("SBIN.NS", start="2015-01-01", end="2024-01-01")
hdfc = yf.download("HDFCBANK.NS", start="2015-01-01", end="2024-01-01")
icici = yf.download("ICICIBANK.NS", start="2015-01-01", end="2024-01-01")
axis = yf.download("AXISBANK.NS", start="2015-01-01", end="2024-01-01")
kotak = yf.download("KOTAKBANK.NS", start="2015-01-01", end="2024-01-01")

sbi.to_csv("sbi_stock_data.csv")
hdfc.to_csv("hdfc_stock_data.csv")
icici.to_csv("icici_stock_data.csv")
axis.to_csv("axis_stock_data.csv")
kotak.to_csv("kotak_stock_data.csv")


