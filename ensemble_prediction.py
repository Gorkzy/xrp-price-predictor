def ensemble_prediction(arima_pred, lstm_pred, weight_arima=0.5, weight_lstm=0.5):
    """
    Kombinuje předpovědi z ARIMA a LSTM pomocí váženého průměru.
    """
    return weight_arima * arima_pred + weight_lstm * lstm_pred

if __name__ == '__main__':
    # Ukázka použití:
    arima_pred = 2.5
    lstm_pred = 2.6
    final_pred = ensemble_prediction(arima_pred, lstm_pred)
    print("Ensemble predikce:", final_pred)
    