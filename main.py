import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import datetime

tf.random.set_seed(42)
np.random.seed(42)


def load_and_preprocess_data(data_path, sample_rate=6):
    """
    加载并预处理气候数据
    参数:
        data_path: 数据文件路径
        sample_rate: 采样率，原始数据为10分钟一次，设置为6表示每小时一次
    返回:
        train_df, val_df, test_df: 训练集、验证集、测试集
        scaler: 归一化器
    """

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"数据文件不存在，请检查路径: {data_path}")

        print("使用模拟数据进行演示...")
        date_rng = pd.date_range(start='2009-01-01', end='2016-12-31', freq='H')
        n = len(date_rng)
        df = pd.DataFrame({
            'Date Time': date_rng,
            'T (degC)': np.sin(np.linspace(0, 20 * np.pi, n)) * 15 + 10 + np.random.normal(0, 2, n),
            'p (mbar)': np.random.normal(1013, 10, n),
            'rh (%)': np.random.normal(60, 15, n),
            'VPmax (mbar)': np.random.normal(17, 3, n),
            'sh (g/kg)': np.random.normal(6, 2, n),
            'H2OC (mmol/mol)': np.random.normal(12, 4, n),
            'rho (g/m**3)': np.random.normal(1200, 50, n),
            'wv (m/s)': np.abs(np.random.normal(3, 2, n)),
            'max. wv (m/s)': np.abs(np.random.normal(5, 3, n)),
            'wd (deg)': np.random.uniform(0, 360, n)
        })


    df = df[sample_rate - 1::sample_rate]
    df.reset_index(drop=True, inplace=True)

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')


    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)


    wd_rad = np.deg2rad(df.pop('wd (deg)'))
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    timestamp_s = (date_time - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    day = 24 * 60 * 60
    year = (365.2425) * day
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


    train_size = int(0.7 * len(df))
    val_size = int(0.2 * len(df))
    test_size = len(df) - train_size - val_size
    train_df, val_df, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]


    features = df.columns.tolist()


    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    print(f"数据加载完成: 训练集={len(train_df)}, 验证集={len(val_df)}, 测试集={len(test_df)}")
    print(f"特征: {features}")

    return train_scaled, val_scaled, test_scaled, scaler, features


def create_sequences(data, window_size=168, forecast_horizon=24, step=1):
    """
    创建时间序列数据集
    参数:
        data: 输入数据
        window_size: 历史窗口大小
        forecast_horizon: 预测未来时间步长
        step: 步长，用于创建更多样本
    返回:
        X, y: 特征和标签
    """
    X, y = [], []
    for i in range(0, len(data) - window_size - forecast_horizon, step):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + forecast_horizon, 0])  # 预测温度（第一列）
    return np.array(X), np.array(y)


def build_rnn_model(input_shape, output_units=24):
    """构建RNN模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru_model(input_shape, output_units=24):
    """构建GRU模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lstm_model(input_shape, output_units=24):
    """构建LSTM模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



def inverse_transform_predictions(y_true, y_pred, scaler):
    """
    将归一化的预测结果反归一化
    """

    y_true_full = np.zeros((len(y_true), scaler.n_features_in_))
    y_pred_full = np.zeros((len(y_pred), scaler.n_features_in_))

    y_true_full[:, 0] = y_true.flatten()
    y_pred_full[:, 0] = y_pred.flatten()


    y_true_inv = scaler.inverse_transform(y_true_full)[:, 0].reshape(y_true.shape)
    y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0].reshape(y_pred.shape)

    return y_true_inv, y_pred_inv


# 评估模型
def evaluate_model(model, X_test, y_test, scaler):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    y_true_inv, y_pred_inv = inverse_transform_predictions(y_test, y_pred, scaler)


    mse = np.mean(mean_squared_error(y_true_inv, y_pred_inv, multioutput='raw_values'))
    mae = np.mean(mean_absolute_error(y_true_inv, y_pred_inv, multioutput='raw_values'))


    overall_mse = mean_squared_error(y_true_inv.flatten(), y_pred_inv.flatten())
    overall_mae = mean_absolute_error(y_true_inv.flatten(), y_pred_inv.flatten())

    return {
        'mse': mse,
        'mae': mae,
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'y_true': y_true_inv,
        'y_pred': y_pred_inv
    }


def visualize_results(models, histories, test_data, scaler, features, window_size=168, forecast_horizon=24):
    """可视化模型训练历史和预测结果"""
    X_test, y_test = test_data

    plt.figure(figsize=(12, 6))
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} Train Loss')
        plt.plot(history.history['val_loss'], label=f'{name} Val Loss')

    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()


    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, scaler)

    metrics_df = pd.DataFrame({
        name: {
            'MSE': result['mse'],
            'MAE': result['mae'],
            'Overall MSE': result['overall_mse'],
            'Overall MAE': result['overall_mae']
        }
        for name, result in results.items()
    })

    print("\n模型评估结果:")
    print(metrics_df)


    plt.figure(figsize=(15, 8))
    sample_idx = np.random.randint(0, len(X_test))
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(len(results), 1, i)
        plt.plot(result['y_true'][sample_idx], 'b-', label='Actual')
        plt.plot(result['y_pred'][sample_idx], 'r-', label='Predicted')
        plt.title(f'{name} - Temperature Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('temperature_prediction_comparison.png')
    plt.close()


    plt.figure(figsize=(12, 6))
    x = np.arange(forecast_horizon)
    width = 0.8 / len(results)

    for i, (name, result) in enumerate(results.items()):
        mae_per_step = mean_absolute_error(result['y_true'], result['y_pred'], multioutput='raw_values')
        plt.bar(x + i * width - 0.4 + width / 2, mae_per_step, width, label=name)

    plt.title('MAE per Forecast Time Step')
    plt.xlabel('Forecast Time Step')
    plt.ylabel('MAE (°C)')
    plt.xticks(x)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('mae_per_time_step.png')
    plt.close()


def main():

    config = {
        'data_path': 'jena_climate_2009_2016.csv',
        'window_size': 168,
        'forecast_horizon': 24,
        'batch_size': 32,
        'epochs': 10,
        'sample_rate': 6
    }


    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
    train_data, val_data, test_data, scaler, features = load_and_preprocess_data(
        config['data_path'],
        sample_rate=config['sample_rate']
    )


    X_train, y_train = create_sequences(
        train_data,
        window_size=config['window_size'],
        forecast_horizon=config['forecast_horizon']
    )
    X_val, y_val = create_sequences(
        val_data,
        window_size=config['window_size'],
        forecast_horizon=config['forecast_horizon']
    )
    X_test, y_test = create_sequences(
        test_data,
        window_size=config['window_size'],
        forecast_horizon=config['forecast_horizon']
    )

    print(f"数据集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"           X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"           X_test={X_test.shape}, y_test={y_test.shape}")


    input_shape = (config['window_size'], len(features))
    models = {
        'RNN': build_rnn_model(input_shape, config['forecast_horizon']),
        'GRU': build_gru_model(input_shape, config['forecast_horizon']),
        'LSTM': build_lstm_model(input_shape, config['forecast_horizon'])
    }


    histories = {}
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/{name}_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=f'logs/{name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            callbacks=callbacks,
            verbose=1
        )

        histories[name] = history

        model.save(f'models/{name}_model_final.h5')


    visualize_results(models, histories, (X_test, y_test), scaler, features,
                      config['window_size'], config['forecast_horizon'])



if __name__ == "__main__":
    main()