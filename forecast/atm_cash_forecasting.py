# forecast/views.py
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import io
import urllib, base64

def data_analysis_view(request):
    # Your existing code
    ds = pd.read_csv('/home/praba/Desktop/ATMcashForecasting/dataFile/AggregatedData.csv')

    ds['Transaction Date'] = pd.to_datetime(ds['Transaction Date'], format='mixed', dayfirst=True)
    ds['Year'] = ds['Transaction Date'].dt.year
    ds['Month'] = ds['Transaction Date'].dt.month
    ds['Day'] = ds['Transaction Date'].dt.day
    ds['DayOfWeek'] = ds['Transaction Date'].dt.dayofweek

    X = ds[['Year', 'Month', 'Day', 'DayOfWeek']]
    y = ds['Total amount Withdrawn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=4, learning_rate=0.05)
    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_test)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(X_test.index, y_test, label='Actual', color='lightgreen')
    plt.plot(X_test.index, y_pred, label='Predicted', color='red')
    plt.xlabel('Transaction Date')
    plt.ylabel('Total amount Withdrawn')
    plt.title('Actual vs Predicted Total Amount Withdrawn')
    plt.legend()
    plt.xticks(rotation=45)

    # Save plot to a PNG image and convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    image_uri = 'data:image/png;base64,' + urllib.parse.quote(image_base64)

    return render(request, 'data_analysis.html', {'image': image_uri})
