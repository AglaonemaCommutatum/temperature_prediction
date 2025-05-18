import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class TemperaturePredictionReport:
    def __init__(self, config=None):
        # 默认配置
        self.config = {
            'data_path': 'jena_climate_2009_2016.csv',
            'models_dir': 'models',
            'results_dir': 'report_results',
            'window_size': 168,
            'forecast_horizon': 24,
            'sample_rate': 6,
            'batch_size': 32,
            'test_samples': 5,
            'metrics_precision': 4,
            'ignore_font_warnings': True
        }


        if config:
            self.config.update(config)

        if not os.path.exists(self.config['results_dir']):
            os.makedirs(self.config['results_dir'])

        self.scaler = None
        self.features = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """加载并预处理数据"""
        try:
            df = pd.read_csv(self.config['data_path'])
        except FileNotFoundError:
            print(f"数据文件不存在，请检查路径: {self.config['data_path']}")
            return None, None, None

        df = df[self.config['sample_rate'] - 1::self.config['sample_rate']]
        df.reset_index(drop=True, inplace=True)


        date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')


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
        _, _, test_df = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]


        self.features = df.columns.tolist()


        self.scaler = MinMaxScaler()
        self.scaler.fit(df[:train_size].values)  # 仅使用训练数据拟合scaler
        test_scaled = self.scaler.transform(test_df)

        print(f"测试数据加载完成: {len(test_df)} 样本")
        return test_df, test_scaled, date_time[train_size + val_size:]

    def create_sequences(self, data):
        """创建时间序列数据集"""
        X, y = [], []
        for i in range(0, len(data) - self.config['window_size'] - self.config['forecast_horizon']):
            X.append(data[i:i + self.config['window_size']])
            y.append(
                data[i + self.config['window_size']:i + self.config['window_size'] + self.config['forecast_horizon'],
                0])
        return np.array(X), np.array(y)

    def load_models(self, model_names=['RNN', 'GRU', 'LSTM']):
        """加载已训练的模型"""
        test_df, test_scaled, _ = self.load_data()
        if test_df is None:
            return None, None

        X_test, y_test = self.create_sequences(test_scaled)
        print(f"测试序列创建完成: {len(X_test)} 样本")

        for name in model_names:
            model_path = os.path.join(self.config['models_dir'], f'{name}_model.h5')
            if os.path.exists(model_path):
                try:
                    self.models[name] = tf.keras.models.load_model(model_path)
                    print(f"{name} 模型加载成功")
                except Exception as e:
                    print(f"加载 {name} 模型失败: {e}")
            else:
                print(f"{name} 模型不存在: {model_path}")

        return X_test, y_test

    def inverse_transform_predictions(self, y_true, y_pred):
        """将预测结果反归一化"""

        print(f"反归一化输入形状: y_true={y_true.shape}, y_pred={y_pred.shape}")

        n_samples, forecast_horizon = y_true.shape


        y_true_full = np.zeros((n_samples * forecast_horizon, self.scaler.n_features_in_))
        y_pred_full = np.zeros((n_samples * forecast_horizon, self.scaler.n_features_in_))

        y_true_full[:, 0] = y_true.flatten()
        y_pred_full[:, 0] = y_pred.flatten()


        y_true_inv = self.scaler.inverse_transform(y_true_full)[:, 0]
        y_pred_inv = self.scaler.inverse_transform(y_pred_full)[:, 0]

        y_true_inv = y_true_inv.reshape(n_samples, forecast_horizon)
        y_pred_inv = y_pred_inv.reshape(n_samples, forecast_horizon)

        print(f"反归一化输出形状: y_true_inv={y_true_inv.shape}, y_pred_inv={y_pred_inv.shape}")
        return y_true_inv, y_pred_inv

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""

        if np.isnan(y_true).any() or np.isnan(y_pred).any():
            print("警告: 预测值或实际值中包含NaN")

            y_true = np.nan_to_num(y_true)
            y_pred = np.nan_to_num(y_pred)

        metrics = {
            'mse': mean_squared_error(y_true.flatten(), y_pred.flatten()),
            'rmse': np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())),
            'mae': mean_absolute_error(y_true.flatten(), y_pred.flatten()),
            'mape': np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100,  # 避免除以0
            'r2': r2_score(y_true.flatten(), y_pred.flatten())
        }


        for key, value in metrics.items():
            metrics[key] = round(value, self.config['metrics_precision'])

        return metrics

    def evaluate_models(self, X_test, y_test):
        """评估所有模型"""
        for name, model in self.models.items():
            print(f"\n评估 {name} 模型...")


            y_pred = model.predict(X_test, verbose=1)
            print(f"{name} 模型输出形状: {y_pred.shape}")


            if y_pred.ndim == 3 and y_pred.shape[2] == 1:
                print(f"警告: {name} 模型输出形状为 {y_pred.shape}，将其reshape为 {y_pred.shape[0], y_pred.shape[1]}")
                y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])


            y_true_inv, y_pred_inv = self.inverse_transform_predictions(y_test, y_pred)


            metrics = self.calculate_metrics(y_true_inv, y_pred_inv)

            step_metrics = {
                'mae': [],
                'rmse': []
            }
            for step in range(self.config['forecast_horizon']):
                step_mae = mean_absolute_error(y_true_inv[:, step], y_pred_inv[:, step])
                step_rmse = np.sqrt(mean_squared_error(y_true_inv[:, step], y_pred_inv[:, step]))
                step_metrics['mae'].append(round(step_mae, self.config['metrics_precision']))
                step_metrics['rmse'].append(round(step_rmse, self.config['metrics_precision']))


            self.results[name] = {
                'metrics': metrics,
                'step_metrics': step_metrics,
                'y_true': y_true_inv,
                'y_pred': y_pred_inv
            }

            print(f"{name} 模型评估完成")

        return self.results

    def generate_plots(self):
        """生成可视化图表"""
        if not self.results:
            print("没有评估结果，无法生成图表")
            return

        plt.figure(figsize=(12, 8))


        plt.subplot(2, 2, 1)
        model_names = list(self.results.keys())
        mae_values = [self.results[name]['metrics']['mae'] for name in model_names]
        plt.bar(model_names, mae_values, color='skyblue')
        plt.title('模型MAE对比')
        plt.ylabel('MAE (°C)')
        for i, v in enumerate(mae_values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')


        plt.subplot(2, 2, 2)
        rmse_values = [self.results[name]['metrics']['rmse'] for name in model_names]
        plt.bar(model_names, rmse_values, color='lightgreen')
        plt.title('模型RMSE对比')
        plt.ylabel('RMSE (°C)')
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')


        plt.subplot(2, 2, 3)
        r2_values = [self.results[name]['metrics']['r2'] for name in model_names]
        plt.bar(model_names, r2_values, color='salmon')
        plt.title('模型R²对比')
        plt.ylabel('R²')
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

        plt.subplot(2, 2, 4)
        mape_values = [self.results[name]['metrics']['mape'] for name in model_names]
        plt.bar(model_names, mape_values, color='gold')
        plt.title('模型MAPE对比')
        plt.ylabel('MAPE (%)')
        for i, v in enumerate(mape_values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['results_dir'], 'model_metrics_comparison.png'))
        plt.close()


        plt.figure(figsize=(14, 6))
        for name, result in self.results.items():
            plt.plot(result['step_metrics']['mae'], label=f'{name} MAE')
            plt.plot(result['step_metrics']['rmse'], '--', label=f'{name} RMSE')

        plt.title('预测步长与误差关系')
        plt.xlabel('预测步长（小时）')
        plt.ylabel('误差 (°C)')
        plt.xticks(range(self.config['forecast_horizon']))
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.config['results_dir'], 'error_vs_forecast_step.png'))
        plt.close()


        for i in range(min(self.config['test_samples'], len(self.results[list(self.results.keys())[0]]['y_true']))):
            plt.figure(figsize=(15, 8))
            for j, (name, result) in enumerate(self.results.items(), 1):
                plt.subplot(len(self.results), 1, j)
                plt.plot(result['y_true'][i], 'b-', label='实际温度')
                plt.plot(result['y_pred'][i], 'r-', label='预测温度')
                plt.title(f'{name} - 样本 {i + 1} 的温度预测')
                plt.xlabel('时间步长')
                plt.ylabel('温度 (°C)')
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.config['results_dir'], f'sample_{i + 1}_prediction.png'))
            plt.close()


        plt.figure(figsize=(12, 12))
        for j, (name, result) in enumerate(self.results.items(), 1):
            plt.subplot(2, 2, j)
            plt.scatter(result['y_true'].flatten(), result['y_pred'].flatten(), alpha=0.3)
            max_val = max(result['y_true'].max(), result['y_pred'].max())
            min_val = min(result['y_true'].min(), result['y_pred'].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.title(f'{name} - 预测值 vs 实际值')
            plt.xlabel('实际温度 (°C)')
            plt.ylabel('预测温度 (°C)')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['results_dir'], 'prediction_vs_actual.png'))
        plt.close()


    def generate_report_data(self):
        """生成报告数据"""
        if not self.results:
            print("没有评估结果，无法生成报告")
            return


        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_metrics': {},
            'step_metrics': {},
            'config': self.config
        }
        for name, result in self.results.items():
            report_data['model_metrics'][name] = result['metrics']


        for name, result in self.results.items():
            report_data['step_metrics'][name] = {
                'mae': result['step_metrics']['mae'],
                'rmse': result['step_metrics']['rmse']
            }


        with open(os.path.join(self.config['results_dir'], 'report_metrics.json'), 'w') as f:
            json.dump(report_data, f, indent=4)

        metrics_df = pd.DataFrame(report_data['model_metrics']).T
        metrics_df.to_csv(os.path.join(self.config['results_dir'], 'model_metrics.csv'))
        for name, metrics in report_data['step_metrics'].items():
            step_df = pd.DataFrame({
                'forecast_step': range(1, self.config['forecast_horizon'] + 1),
                'mae': metrics['mae'],
                'rmse': metrics['rmse']
            })
            step_df.to_csv(os.path.join(self.config['results_dir'], f'{name}_step_metrics.csv'), index=False)

        print(f"报告数据已保存至 {self.config['results_dir']}")
        return report_data


def main():

    config = {
        'data_path': 'jena_climate_2009_2016.csv',
        'models_dir': 'models',
        'results_dir': 'report_results',
        'window_size': 168,
        'forecast_horizon': 24,
        'sample_rate': 6,
        'test_samples': 5,
        'metrics_precision': 4,
        'ignore_font_warnings': True
    }


    report_generator = TemperaturePredictionReport(config)


    X_test, y_test = report_generator.load_models(['RNN', 'GRU', 'LSTM'])

    if X_test is not None:

        report_generator.evaluate_models(X_test, y_test)

        report_generator.generate_plots()


        report_data = report_generator.generate_report_data()


        print("\n模型评估主要指标:")
        for name, metrics in report_data['model_metrics'].items():
            print(f"\n{name} 模型:")
            print(f"  MAE:  {metrics['mae']:.4f} °C")
            print(f"  RMSE: {metrics['rmse']:.4f} °C")
            print(f"  R²:   {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.4f} %")




if __name__ == "__main__":
    main()