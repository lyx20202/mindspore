import os
import numpy as np
import pandas as pd
from datetime import datetime
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.train.callback import LossMonitor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 用于显示进度条

# ---------------------------
# 1. 数据预处理函数
# ---------------------------
def preprocess_data(csv_path, window_size=24):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 将日期时间转换为datetime对象，并按时间排序
    df['date_time'] = pd.to_datetime(df['date_time'], format='%Y/%m/%d %H:%M')
    df = df.sort_values('date_time').reset_index(drop=True)
    
    # 提取时间特征：小时和星期几
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek

    # 数值特征
    num_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'dayofweek']
    scaler_num = StandardScaler()
    df[num_features] = scaler_num.fit_transform(df[num_features])
    
    # 对天气类别进行One-Hot编码
    cat_features = ['weather_main', 'weather_description']
    encoder = OneHotEncoder(sparse_output=False)
    weather_encoded = encoder.fit_transform(df[cat_features])
    weather_encoded_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(cat_features))
    df = pd.concat([df, weather_encoded_df], axis=1)
    df.drop(columns=cat_features, inplace=True)

    # 整体特征列表（不包含目标 traffic_volume 与日期字段）
    feature_cols = num_features + list(weather_encoded_df.columns)
    
    # 对目标 traffic_volume 进行标准化处理
    target_col = 'traffic_volume'
    scaler_target = StandardScaler()
    df[target_col] = scaler_target.fit_transform(df[[target_col]])

    # 使用滑动窗口方式构造数据：用连续 window_size 时刻预测下一个时刻值
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[feature_cols].iloc[i:i+window_size].values)
        y.append(df[target_col].iloc[i+window_size])
    X = np.array(X, dtype=np.float32)    # 形状：(样本数, window_size, 特征数)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)  # 形状：(样本数, 1)
    
    return X, y, scaler_num, scaler_target

# ---------------------------
# 2. 构造MindSpore数据集
# ---------------------------
def dataset_generator(X, y):
    for i in range(len(X)):
        yield X[i], y[i]

def create_ms_dataset(X, y, batch_size=32, shuffle=True):
    data_gen = lambda: dataset_generator(X, y)
    dataset = ds.GeneratorDataset(source=data_gen, column_names=["features", "target"])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

# ---------------------------
# 3. 定义 MSTIM 模型（LSTM-CNN-Attention）
# ---------------------------
# 修改后的 MSTIM 模型定义
class MSTIM(nn.Cell):
    def __init__(self, input_dim, lstm_hidden_size=64, lstm_layers=1, cnn_filters=64, kernel_size=3, attention_size=32):
        super(MSTIM, self).__init__()
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_layers,
                            batch_first=True)
        
        # CNN 层：多尺度卷积
        self.convs = nn.CellList([
            nn.Conv1d(
                in_channels=lstm_hidden_size, 
                out_channels=cnn_filters, 
                kernel_size=scale, 
                stride=1, 
                pad_mode='pad', 
                padding=scale//2
            ) 
            for scale in [3, 5, 7]  # 使用不同尺度的卷积核
        ])
        self.relu = nn.ReLU()
        
        # Attention 层
        # 合并后的通道数为 cnn_filters * 3（三个卷积层）
        self.attention_weight = nn.Dense(cnn_filters * 3, attention_size)
        self.attention_context = nn.Dense(attention_size, 1)

        # 全连接层
        self.fc = nn.Dense(cnn_filters * 3, 1)

    def construct(self, x):
        # x 形状: (batch_size, seq_len, input_dim)
        # LSTM 层
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_hidden_size)
        
        # CNN 层处理多尺度特征
        lstm_out = lstm_out.transpose(0, 2, 1)  # 转换为 (batch_size, lstm_hidden_size, seq_len)
        cnn_outs = []
        for conv in self.convs:
            c_out = conv(lstm_out)     # 每个卷积输出形状: (batch_size, cnn_filters, seq_len)
            cnn_outs.append(c_out)
        # 合并不同卷积核的输出（沿通道维度拼接）
        cnn_out = ops.concat(cnn_outs, axis=1)  # 形状: (batch_size, cnn_filters*3, seq_len)
        cnn_out = self.relu(cnn_out)
        
        # 调整维度用于注意力计算
        cnn_out = cnn_out.transpose(0, 2, 1)  # 转换为 (batch_size, seq_len, cnn_filters*3)
        
        # Attention 层
        attention_scores = self.attention_weight(cnn_out)  # (batch_size, seq_len, attention_size)
        attention_scores = ops.Softmax(axis=1)(attention_scores)  # 对时间步进行归一化
        # 计算加权和
        attention_output = ops.matmul(attention_scores.transpose(0, 2, 1), cnn_out)  # (batch_size, attention_size, cnn_filters*3)
        attention_output = attention_output.sum(axis=1)  # 聚合得到 (batch_size, cnn_filters*3)
        
        # 全连接层预测
        output = self.fc(attention_output)
        return output
# ---------------------------
# 4. 训练流程（带进度条）
# ---------------------------
def train_model(model, dataset, num_epochs=10, learning_rate=0.001):
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    net_with_loss = nn.WithLossCell(model, loss_fn)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        step = 0
        steps = dataset.get_dataset_size()
        # 使用 tqdm 进度条显示每个 epoch 的 batch 进度
        for data in tqdm(dataset.create_tuple_iterator(), total=steps, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features, target = data
            loss = train_network(features, target)
            epoch_loss += loss.asnumpy()
            step += 1
        avg_loss = epoch_loss / step
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return model

# ---------------------------
# 5. 模型评估函数（仅包含时间序列回归评价指标）
# ---------------------------
def evaluate_model(model, dataset):
    model.set_train(False)
    predictions = []
    true_values = []
    steps = dataset.get_dataset_size()
    for data in tqdm(dataset.create_tuple_iterator(), total=steps, desc="Evaluating"):
        features, target = data
        pred = model(features)
        predictions.extend(pred.asnumpy().flatten().tolist())
        true_values.extend(target.asnumpy().flatten().tolist())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # 回归指标
    mse = np.mean(np.square(predictions - true_values))
    mae = np.mean(np.abs(predictions - true_values))
    rmse = np.sqrt(mse)
    print("\n【回归评价指标】")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return mse, mae, rmse

# ---------------------------
# 6. 主函数：数据加载、模型训练与评估
# ---------------------------
def main():
    csv_path = "Traffic.csv"  # 数据文件路径
    window_size = 24  # 用过去24小时数据预测下一时刻交通流量
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # 数据预处理
    X, y, scaler_features, scaler_target = preprocess_data(csv_path, window_size=window_size)
    print("预处理后数据形状：", X.shape, y.shape)  # 例如：(样本数, 24, 特征数)
    
    # 直接划分训练集与测试集（80%训练，20%测试），不再使用分层抽样
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构造MindSpore数据集
    train_dataset = create_ms_dataset(train_X, train_y, batch_size=batch_size, shuffle=True)
    test_dataset  = create_ms_dataset(test_X, test_y, batch_size=batch_size, shuffle=False)
    
    # 输入特征数量
    input_dim = X.shape[-1]
    
    # 构建交通流量预测模型
    model = MSTIM(input_dim=input_dim, lstm_hidden_size=64, lstm_layers=1, cnn_filters=64, kernel_size=3, attention_size=32)
    
    # 模型训练（带进度条）
    trained_model = train_model(model, train_dataset, num_epochs=num_epochs, learning_rate=learning_rate)
    
    # 模型评估（仅显示回归指标，并显示进度条）
    print("\n开始在测试集上评估模型：")
    evaluate_model(trained_model, test_dataset)
    
    # 保存模型参数
    ms.save_checkpoint(trained_model, "traffic_flow_predictor.ckpt")
    print("模型训练与评估完成，并已保存模型。")

if __name__ == "__main__":
    main()
# Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:34<00:00,  7.79it/s]
# Epoch [1/10], Loss: 0.3361
# Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:34<00:00,  7.79it/s]
# Epoch [1/10], Loss: 0.3361
# Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:39<00:00,  7.55it/s]
# Epoch [2/10], Loss: 0.1723
# Epoch 2/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:39<00:00,  7.55it/s]
# Epoch [2/10], Loss: 0.1723
# Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:37<00:00,  7.63it/s]
# Epoch [3/10], Loss: 0.1350
# Epoch 3/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:37<00:00,  7.63it/s]
# Epoch [3/10], Loss: 0.1350
# Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:38<00:00,  7.60it/s]
# Epoch [4/10], Loss: 0.1167
# Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:34<00:00,  7.78it/s]
# Epoch [5/10], Loss: 0.1066
# Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:34<00:00,  7.78it/s]
# Epoch [5/10], Loss: 0.1066
# Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:39<00:00,  7.55it/s]
# Epoch [6/10], Loss: 0.0997
# Epoch 6/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:39<00:00,  7.55it/s]
# Epoch [6/10], Loss: 0.0997
# Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:40<00:00,  7.48it/s]
# Epoch [7/10], Loss: 0.0955
# Epoch 7/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:40<00:00,  7.48it/s]
# Epoch [7/10], Loss: 0.0955
# Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:40<00:00,  7.50it/s]
# Epoch [8/10], Loss: 0.0903
# Epoch 8/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:40<00:00,  7.50it/s]
# Epoch [8/10], Loss: 0.0903
# Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:41<00:00,  7.48it/s]
# Epoch [9/10], Loss: 0.0870
# Epoch 10/10:  24%|██████████████████▋                                                         Epoch 10/10:  24%|██████████████████▊                                                         Epoch 10/10:  24%|██████████████████▉                                                         Epoch 10/10:  24%|██████████████████▉                                                         Epoch 10/10:  24%|███████████████████                                                         Epoch 10/10:  25%|███████████████████                                                         Epoch 10/10:  25%|███████████████████▏                                                        Epoch 10/10:  25%|███████████████████▏                                                        Epoch 10/10:  25%|███████████████████▎                                                        Epoch 10/10:  25%|███████████████████▍                                                        Epoch 10/10:  25%|███████████████████▌                                                        Epoch 10/10:  25%|███████████████████▋                                                        Epoch 10/10:  25%|███████████████████▋                                                        Epoch 10/10:  25%|███████████████████▊                                                        Epoch 10/10:  25%|███████████████████▊                                                        Epoch 10/10:  25%|███████████████████▉                                                        Epoch 10/10:  26%|████████████████████                                                        Epoch 10/10:  26%|████████████████████                                                        Epoch 10/10:  26%|████████████████████▏                                                       Epoch 10/10:  26%|████████████████████▏                                                       Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:41<00:00,  7.48it/s]
# Epoch [9/10], Loss: 0.0870
# Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████| 1204/1204 [02:37<00:00,  7.65it/s]
# Epoch [10/10], Loss: 0.0839

# 开始在测试集上评估模型：
# Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████| 301/301 [00:12<00:00, 23.67it/s]

# 【回归评价指标】
#   MAE : 0.2120
#   MSE : 0.1048
#   RMSE: 0.3237
# 模型训练与评估完成，并已保存模型。