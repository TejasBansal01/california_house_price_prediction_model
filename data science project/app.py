import streamlit as st
#title
import pandas as pd
import random
import pickle
from sklearn.preprocessing import StandardScaler
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('california Housing Price Prediction')

st.image('https://thf.bing.com/th/id/OIP.kFNEzzGa5lvZ-L-KyPJCSAHaFl?w=177&h=180&c=7&r=0&o=7&cb=thfc1&dpr=1.3&pid=1.7&rm=3')

st.header('model of housing prices to predict median house values in california',divider=True)
#st.subheader('''user must enter given values to predict price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')
st.sidebar.title('select house features')
st.sidebar.image('https://www.bing.com/th/id/OIP.vEC5VU62eJutBXsO6ZUt_gHaEJ?w=245&h=211&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2')
temp_df=pd.read_csv('california.csv')
random.seed(12)
all_values=[]
for i in temp_df[col]:
    min_value,max_value=temp_df[i].agg(['min','max'])
    var=st.sidebar.slider(f'select {i} range',int(min_value),int(max_value),random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])
import pickle
with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt=pickle.load(f)

price=chatgpt.predict(final_value)[0]

import time

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('predicting price')

place = st.empty()
place.image('https://www.bing.com/th/id/OIP.fe6oDzj36om7NH_ddPovGgHaFj?w=258&h=211&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2',width=100)

if price>0:
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i+1)

    
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    # st.subheader(body)

    st.success(body)
else:
    body='invalid house feature'
    st.warning(body)