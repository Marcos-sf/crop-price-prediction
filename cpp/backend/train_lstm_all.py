import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

mandis = {
    'arecanut': ['Sirsi', 'Yellapur', 'Siddapur', 'Shimoga', 'Sagar', 'Kumta'],
    'coconut': ['Bangalore', 'Arasikere', 'Channarayapatna', 'Ramanagara', 'Sira', 'Tumkur']
}

for crop, mandi_list in mandis.items():
    for mandi in mandi_list:
        features_path = f"app/data/processed/features_{crop}_{mandi.lower()}.npy"
        target_path = f"app/data/processed/target_{crop}_{mandi.lower()}.npy"
        if not os.path.exists(features_path) or not os.path.exists(target_path):
            print(f"Missing data for {crop.title()} in {mandi}, skipping.")
            continue
        X = np.load(features_path)
        y = np.load(target_path)
        if len(X) < 10:
            print(f"Not enough data for {crop.title()} in {mandi}, skipping.")
            continue
        # Normalize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std
        # Reshape for LSTM: (samples, timesteps, features)
        X_norm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, shuffle=False)
        # Build LSTM model
        model = Sequential([
            LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es], verbose=0)
        y_pred = model.predict(X_test).flatten()
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"LSTM RMSE for {crop.title()} in {mandi.title()}: {rmse:.2f}")
        # Save model and normalization params
        model_path = f"app/data/processed/lstm_{crop}_{mandi.lower()}.h5"
        norm_path = f"app/data/processed/lstm_norm_{crop}_{mandi.lower()}.joblib"
        model.save(model_path)
        joblib.dump({'mean': X_mean, 'std': X_std}, norm_path)
        print(f"Saved LSTM model to {model_path} and normalization to {norm_path}\n") 