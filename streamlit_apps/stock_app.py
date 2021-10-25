import streamlit as st
import pandas as pd
import yfinance as yf

st.write(
    """
    ## Simple Stock Price App 
    
    Shown are the stock closing price and volume of Google!
    
    """
)

ticker = 'GOOGL'
tickerData = yf.Ticker(ticker)

tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)