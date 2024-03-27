import joblib
import numpy as np
import sam_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import optuna, os

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/Data-for-Data-Mining-Project - Source.csv')

    # Bỏ nhãn "Spam"
    data = data[~data['Label'].isin(['Spam'])].reset_index(drop=True)

    # Chuyển Label về dạng nhị phân, với Positive: 1; Negative: 0.
    data['Label'] = data['Label'].isin(['Positive']).astype(int)

    # Chia tập train, test, validate
    train_test_val_index = np.array(range(data.shape[0]))
    train_index = np.random.choice(train_test_val_index, int(len(train_test_val_index)*0.7), replace=False)

    test_val_index = train_test_val_index[~np.isin(train_test_val_index, train_index)]
    test_index = np.random.choice(test_val_index, int(len(train_test_val_index)*0.2), replace=False)

    val_index = test_val_index[~np.isin(test_val_index, test_index)]

    data_train = data.iloc[train_index]
    data_test = data.iloc[test_index]
    data_val = data.iloc[val_index]
    data_train.shape, data_test.shape, data_val.shape
    
    # Tunning parameters
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: sam_model.objective(trial, data_train, data_test), n_trials=24)

    
    print(f'\nKết quả tốt nhất : {study.best_value}')
    print(f'Bộ tham số của kết quả tốt nhất : {study.best_params}')

    # Chạy lại mô hình với bộ test để thấy các thông số khác của kết quả tốt nhất
    X_train, X_test = sam_model.create_X(data_train, data_test, study.best_params['data-type'])
    y_train, y_test = data_train['Label'], data_test['Label']

    model = sam_model.Sam_Model(study.best_params['model-type'], study.best_params['kernel'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('\nChạy lại mô hình với bộ test để thấy các thông số khác của kết quả tốt nhất:')
    print(f'Accuracy : {accuracy_score(y_test, y_pred):.3f}')
    print(f'Recall : {recall_score(y_test, y_pred):.3f}')
    print(f'Precision : {precision_score(y_test, y_pred):.3f}')
    print(f'F1 Score : {f1_score(y_test, y_pred):.3f}')

    # Đánh giá mô hình bằng bộ val
    X_train, X_val = sam_model.create_X(data_train, data_val, study.best_params['data-type'])
    y_train, y_val = data_train['Label'], data_val['Label']

    model = sam_model.Sam_Model(study.best_params['model-type'], study.best_params['kernel'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print('\nĐánh giá mô hình bằng bộ val:')
    print(f'Accuracy : {accuracy_score(y_val, y_pred):.3f}')
    print(f'Recall : {recall_score(y_val, y_pred):.3f}')
    print(f'Precision : {precision_score(y_val, y_pred):.3f}')
    print(f'F1 Score : {f1_score(y_val, y_pred):.3f}')

    # Lưu mô hình lại, để sau dùng
    os.makedirs('model')
    joblib.dump(model, "model/sam_model.pkl")
    print('\nSave model to model/sam_model.pkl')
    print('Success!')