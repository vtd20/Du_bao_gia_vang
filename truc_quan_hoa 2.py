# truc_quan_hoa
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import year

def plot_gold_price_history(df):
    # Convert to Pandas DataFrame
    pdf = df.toPandas()
    
    # Plotly line plot
    fig = px.line(pdf, x='Date', y='Price')
    fig.update_traces(line_color='blue')
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Scaled Price",
        title={'text': "Gold Price History Data", 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
        plot_bgcolor='rgba(255,223,0,0.8)'
    )
    
    # If running in an interactive environment, display the plot; otherwise, save as HTML
    try:
        fig.show()  # If you're in a Jupyter Notebook or another interactive environment
    except:
        fig.write_html("gold_price_history.html")  # Save as HTML for non-interactive environments

def plot_train_test_split(df, test_size):
    # Convert to Pandas DataFrame
    pdf = df.toPandas()
    
    # Matplotlib plot for train/test split
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='black')
    
    plt.plot(pdf['Date'][:-test_size], pdf['Price'][:-test_size], color='blue', lw=2)
    plt.plot(pdf['Date'][-test_size:], pdf['Price'][-test_size:], color='red', lw=2)
    
    plt.title('Gold Price Training and Test Sets', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 15})
    plt.grid(color='white')
    
    # Save plot as PNG if not in interactive environment
    plt.savefig("train_test_split.png")
    plt.close()  # Close the plot to free up memory

def plot_model_performance(df, train_data, y_test_true, y_test_pred, test_size):
    # Convert to Pandas DataFrame
    pdf = df.toPandas()
    
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='black')
    
    plt.plot(pdf['Date'].iloc[:-test_size], train_data, color='green', lw=2)
    plt.plot(pdf['Date'].iloc[-test_size:], y_test_true, color='blue', lw=2)
    plt.plot(pdf['Date'].iloc[-test_size:], y_test_pred, color='red', lw=2)
    
    plt.title('Model Performance on Gold Price Prediction', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})
    plt.grid(color='white')
    
    # Save plot as PNG if not in interactive environment
    plt.savefig("model_performance.png")
    plt.close()  # Close the plot to free up memory

