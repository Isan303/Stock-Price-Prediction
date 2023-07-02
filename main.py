
import streamlit as st
from datetime import date
from plotly import graph_objs as go
import yfinance as yf
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

stocks = ("HCLTECH.NS",'GOOG', 'AAPL', 'MSFT', 'GME',"RELI","TCS.NS","INFY","WIPRO.NS",
          "LT.NS","LTI.NS","RELIANCE.NS","INFY.NS","INFY.BO","MRF.NS","DRREDDY.NS","NESTLEIND.NS","BAJFINANCE.NS")
selected_stock = st.selectbox('Select dataset for prediction', stocks)



@st.cache
def load_data(ticker):
    df =yf.download(ticker, START,TODAY)
    df.reset_index(inplace=True)
    return df
data_load_state=st.text("Load data ....")
df=load_data(selected_stock)
data_load_state.text("DONE!")

st.subheader('Raw data')
st.write(df.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

	
data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*.8)

scaler =MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data=scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
  if i<=61:
    print(x_train)
    print(y_train)
    print()


x_train,y_train =np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1 ))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,batch_size=1,epochs=1)

test_data=scaled_data[training_data_len-60: , :]
x_test=[]
y_test =dataset[training_data_len:, :]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])
  
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)


train =data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
d=df['Date']

def predictionplot():
    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=d[training_data_len:], y=valid['Predictions'], name="predicted"))
    fig1.add_trace(go.Scatter(x=d[training_data_len:], y=valid['Close'], name="value"))
    fig1.add_trace(go.Scatter(x=df['Date'], y=train['Close'], name="Trainig dataset"))
    fig1.layout.update(title=('Predicted Plot For '+selected_stock) ,xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)
    
predictionplot()

rmse=(np.mean((predictions - y_test)/y_test)*100)
st.subheader('Error percentage')
st.write(rmse)


new_df=df.filter(['Close'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
st.subheader('Predicted price for tomorrow')
st.write(pred_price-((pred_price/100)*rmse))

