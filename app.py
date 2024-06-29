import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    return df

def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Logistic Regression': 
        model = LogisticRegression(random_state=42)
    
    model.fit(X_train_scaled, y_train)
    return model, scaler


def forecast_next_week(model, scaler):
    next_week = pd.DataFrame({
        'DayOfWeek': range(7),
        'Month': [pd.Timestamp.now().month] * 7
    })
    next_week_scaled = scaler.transform(next_week)
    predictions = model.predict(next_week_scaled)
    return predictions


st.set_page_config(
    page_title="VertiClimb",
    page_icon=":person_climbing:", 
)

st.title('VertiClimb')
st.title('Gym Capacity Forecast 🧗‍♂️📈')


# file upload
uploaded_file = st.file_uploader("Upload your gym data:", type="csv")
use_pre_loaded_button = st.button("Or use preloaded demo data")
pre_loaded_file_path = "./gym_capacity_data.csv"

if uploaded_file is not None or use_pre_loaded_button:
    if use_pre_loaded_button:
        data = pd.read_csv(pre_loaded_file_path)
    else:
        data = pd.read_csv(uploaded_file)
    data = preprocess_data(data)
    
    # model selection
    model_type = st.selectbox(
        'Choose a model', 
        ['Random Forest', 'XGBoost', 'Logistic Regression']
        )
    
    # model training
    if st.button('Train Model'):
        X = data[['DayOfWeek', 'Month']]
        y = data['Capacity']
        model, scaler = train_model(X, y, model_type)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.success('Model trained successfully!')
    
    # predictions
    if st.button('Forecast Next Week'):
        if 'model' in st.session_state and 'scaler' in st.session_state:
            predictions = forecast_next_week(st.session_state['model'], st.session_state['scaler'])
        
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
            color_scale = px.colors.sequential.Sunsetdark
            
            # normalize the predictions to a 0-1 scale for color mapping
            norm_predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            
            fig = go.Figure(data=[go.Bar(
                x=days,
                y=predictions,
                marker=dict(
                    color=norm_predictions,
                    colorscale=color_scale,
                    colorbar=dict(title="Occupancy Level")
                )
            )])
            
            fig.update_layout(
                title='Forecasted Gym Capacity for Next Week',
                xaxis_title='Day of Week',
                yaxis_title='Capacity',
            )
            
            st.plotly_chart(fig)    
        else:
            st.warning('Please train the model first!')