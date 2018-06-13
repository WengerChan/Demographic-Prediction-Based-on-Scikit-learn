# 主程序
# import modules
import pandas as pd
import os
from pd_tools import split_train_test, get_part_data
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from ml_tools import get_best_model
from sklearn.metrics import log_loss
from sklearn.feature_selection import VarianceThreshold
from sklearn.externals import joblib
import pickle


dataset_path = './dataset'

gender_age_filename = 'gender_age.csv'
phone_brand_device_model_filename = 'phone_brand_device_model.csv'
events_filename = 'events.csv'
app_events_filename = 'app_events.csv'
app_labels_filename = 'app_labels.csv'
label_categories_filename = 'label_categories.csv'

train_gender_age_filename = 'gender_age_train.csv'
test_gender_age_filename = 'gender_age_test.csv'


def split_dataset(is_first_run=True):
    """
    1.Split dataset
    :param is_first_run = True
    """
    if is_first_run:
        print('-----Split dataset-----')
        all_gender_age = pd.read_csv(os.path.join(dataset_path, gender_age_filename))

        # print(all_gender_age.isnull()) # isnull: missing value

        df_train, df_test = split_train_test(all_gender_age)
        print(df_train.head())
        print(df_test.head())

        print('The number of train set: ', df_train.groupby('group').size())
        print('The number of test set: ', df_test.groupby('group').size())

        # Saving
        df_train.to_csv(os.path.join(dataset_path, train_gender_age_filename), index=False)
        df_test.to_csv(os.path.join(dataset_path, test_gender_age_filename), index=False)

        print("Train set has been saved to 'gender_age_train.csv'.")
        print("Test set has been saved to 'gender_age_test.csv'.")


def load_dataset(percent=0.1):
    """
    2.Load dataset
    :param percent = 0.1
    :return:gender_age_train,
        gender_age_test,
        phone_brand_device_model,
        events,
        app_events
    """
    print('-----Load dataset-----')
    # 'read_csv': read csv to a DataFrame objects
    # index_col: column index
    gender_age_train = \
        pd.read_csv(os.path.join(dataset_path, train_gender_age_filename), index_col='device_id')
    gender_age_test = \
        pd.read_csv(os.path.join(dataset_path, test_gender_age_filename), index_col='device_id')

    gender_age_train = get_part_data(gender_age_train, percent=percent)
    gender_age_test = get_part_data(gender_age_test, percent=percent)
    print("Successfully read 'gender_age_train.csv'.")
    print(gender_age_train.head())
    print("Successfully read 'gender_age_test.csv'.")
    print(gender_age_test.head())

    phone_brand_device_model =\
        pd.read_csv(os.path.join(dataset_path, phone_brand_device_model_filename))
    phone_brand_device_model = \
        phone_brand_device_model.drop_duplicates('device_id').set_index('device_id')  # drop repeat value
    print("Successfully read 'phone_brand_device_model.csv'.")
    print(phone_brand_device_model.head())

    events = pd.read_csv(os.path.join(dataset_path, events_filename),
                         usecols=['device_id', 'event_id'], index_col='event_id')
    # FutureWarning
    # events = pd.read_csv(os.path.join(dataset_path, events_filename),
    #                      usecols=['device_id', 'event_id'])
    print("Successfully read 'events.csv'.")
    print(events.head())

    app_events = pd.read_csv(os.path.join(dataset_path, app_events_filename),
                             usecols=['event_id', 'app_id'])
    print("Successfully read 'app_events.csv'.")
    print(app_events.head())

    # app_labels = pd.read_csv(os.path.join(dataset_path, app_labels_filename))
    # print("Successfully read 'app_labels.csv'.")

    return gender_age_train, gender_age_test, phone_brand_device_model, events, app_events


def feature_engineering(gender_age_train, gender_age_test, phone_brand_device_model, events, app_events):
    """
    3.Feature_engineering
    4.Add label
    :return: tr_feat_scaled_sel_pca, te_feat_scaled_sel_pca, y_train, y_test
    """
    print('-----Feature engineering-----')

    # 3.1 phone_brand
    # LabelEncoder-fit-transform-OneHotEncoder
    brand_label_encoder = LabelEncoder()

    brand_label_encoder.fit(phone_brand_device_model['phone_brand'].values)

    phone_brand_device_model['brand_label_code'] = \
        brand_label_encoder.transform(phone_brand_device_model['phone_brand'].values)

    gender_age_train['brand_label_code'] = phone_brand_device_model['brand_label_code']
    gender_age_test['brand_label_code'] = phone_brand_device_model['brand_label_code']

    brand_onehot_encoder = OneHotEncoder()
    brand_onehot_encoder.fit(phone_brand_device_model['brand_label_code'].values.reshape(-1, 1))  # reshape(-1,1)转成一列

    tr_brand_feat = \
        brand_onehot_encoder.transform(gender_age_train['brand_label_code'].values.reshape(-1, 1))
    te_brand_feat = \
        brand_onehot_encoder.transform(gender_age_test['brand_label_code'].values.reshape(-1, 1))

    print('[phone brand]Feature Dimensions: ', tr_brand_feat.shape[1])

    # 3.2 phone_model: phone_model=phone_brand+device_model
    phone_brand_device_model['brand_model'] = \
        phone_brand_device_model['phone_brand'].str.cat(phone_brand_device_model['device_model'])

    # LabelEncoder-fit-transform-OneHotEncoder
    model_label_encoder = LabelEncoder()

    model_label_encoder.fit(phone_brand_device_model['brand_model'].values)

    phone_brand_device_model['brand_model_label_code'] = \
        model_label_encoder.transform(phone_brand_device_model['brand_model'].values)

    gender_age_train['brand_model_label_code'] = \
        phone_brand_device_model['brand_model_label_code']
    gender_age_test['brand_model_label_code'] = \
        phone_brand_device_model['brand_model_label_code']

    model_onehot_encoder = OneHotEncoder()
    model_onehot_encoder.fit(phone_brand_device_model['brand_model_label_code'].values.reshape(-1, 1))

    tr_model_feat = \
        model_onehot_encoder.transform(gender_age_train['brand_model_label_code'].values.reshape(-1, 1))
    te_model_feat = \
        model_onehot_encoder.transform(gender_age_test['brand_model_label_code'].values.reshape(-1, 1))

    print('[phone model]Feature Dimensions: ', tr_model_feat.shape[1])

    # 3.3 APP
    device_app = app_events.merge(events, how='left', left_on='event_id', right_index=True)
    n_run_s = device_app['app_id'].groupby(device_app['device_id']).size()
    n_app_s = device_app['app_id'].groupby(device_app['device_id']).nunique()

    gender_age_train['n_run'] = n_run_s
    gender_age_train['n_app'] = n_app_s
    # fill missing values
    gender_age_train['n_run'].fillna(0, inplace=True)
    gender_age_train['n_app'].fillna(0, inplace=True)

    gender_age_test['n_run'] = n_run_s
    gender_age_test['n_app'] = n_app_s
    # fill missing values
    gender_age_test['n_run'].fillna(0, inplace=True)
    gender_age_test['n_app'].fillna(0, inplace=True)

    tr_run_feat = gender_age_train['n_run'].values.reshape(-1, 1)
    tr_app_feat = gender_age_train['n_app'].values.reshape(-1, 1)

    te_run_feat = gender_age_test['n_run'].values.reshape(-1, 1)
    te_app_feat = gender_age_test['n_app'].values.reshape(-1, 1)

    # 3.4 Merge all features
    tr_feat = np.hstack((tr_brand_feat.toarray(), tr_model_feat.toarray(),
                         tr_run_feat, tr_app_feat))
    te_feat = np.hstack((te_brand_feat.toarray(), te_model_feat.toarray(),
                         te_run_feat, te_app_feat))

    print('-----Feature extraction Ended-----')
    print('Feature dimensions of each sample: ', tr_feat.shape[1])

    print('tr_feat', '\n', tr_feat[0:5])
    print('te_feat', '\n', te_feat[0:5])

    # 3.5 特征范围归一化
    scaler = StandardScaler()

    tr_feat_scaled = scaler.fit_transform(tr_feat)
    te_feat_scaled = scaler.transform(te_feat)

    print('tr_feat_scaled', '\n', tr_feat_scaled[0:5])
    print('tr_feat_scaled', '\n', te_feat_scaled[0:5])

    # 3.6 特征选择
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

    tr_feat_scaled_sel = sel.fit_transform(tr_feat_scaled)
    te_feat_scaled_sel = sel.transform(te_feat_scaled)

    print('tr_feat_scaled_sel', '\n', tr_feat_scaled[0:5])
    print('tr_feat_scaled-sel', '\n', te_feat_scaled[0:5])

    # 3.7 PCA
    # Principal Component Analysis
    pca = PCA(n_components=0.95)  # save 95%贡献率的特征向量
    tr_feat_scaled_sel_pca = pca.fit_transform(tr_feat_scaled_sel)
    te_feat_scaled_sel_pca = pca.transform(te_feat_scaled_sel)

    print('-----Feature engineering Ended-----')
    print('Feature dimensions of each sample(processed)：', tr_feat_scaled_sel_pca.shape[1])

    print('tr_feat_scaled_sel_pca', '\n', tr_feat_scaled[0:5])
    print('tr_feat_scaled-sel_pca', '\n', te_feat_scaled[0:5])

    # 4 Add label
    # LabelEncoder-fit-transform
    group_label_encoder = LabelEncoder()

    group_label_encoder.fit(gender_age_train['group'].values)

    y_train = group_label_encoder.transform(gender_age_train['group'].values)
    y_test = group_label_encoder.transform(gender_age_test['group'].values)

    # Save
    save_data(1, tr_feat_scaled_sel_pca)
    save_data(2, te_feat_scaled_sel_pca)
    save_data(3, y_train)
    save_data(4, y_test)

    return tr_feat_scaled_sel_pca, te_feat_scaled_sel_pca, y_train, y_test


def save_data(num, data_file):
    """
    Save value data including y_train and y_test
    :param data_file:
    """
    name_dict = {1:"tr_feat_scaled_sel_pca", 2:"te_feat_scaled_sel_pca", 3:"y_train", 4:"y_test"}
    name = name_dict[num] + '.pkl'
    p = open(name, 'wb')
    pickle.dump(data_file, p)
    p.close()


def run_main():
    """
    main function
    1.Split dataset
    2.Load dataset
    3.Feature_engineering
    4.Add label
    """
    # 1.Split dataset
    split_dataset(True)

    # 2.Load dataset
    gender_age_train, gender_age_test, phone_brand_device_model, events, app_events = load_dataset(percent=0.3)

    # 3.Feature_engineering
    # 4.Add label
    tr_feat_scaled_sel_pca, te_feat_scaled_sel_pca, y_train, y_test = \
        feature_engineering(gender_age_train, gender_age_test, phone_brand_device_model, events, app_events)

    # 5.Train and Test model
    print('-----01.Logistic Regression-----')
    lr_param_grid = [
        {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    ]
    lr_model = LogisticRegression()
    best_lr_model = get_best_model(lr_model,
                                   tr_feat_scaled_sel_pca, y_train,
                                   lr_param_grid, cv=3)  # 3-fold Cross Validation
    joblib.dump(best_lr_model, 'best_lr_model.m')  # save model
    y_pred_lr = best_lr_model.predict_proba(te_feat_scaled_sel_pca)
    print("Logistic Regression's logloss: ", log_loss(y_test, y_pred_lr))

    print('-----02.Train SVM-----')
    svm_param_grid = [
        {'C': [1e-2, 1e-1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svm_model = svm.SVC(probability=True)  # probability默认False为二分类；设为True可做多分类
    best_svm_model = get_best_model(svm_model,
                                    tr_feat_scaled_sel_pca, y_train,
                                    svm_param_grid, cv=3)  # 3-fold Cross Validation
    joblib.dump(best_svm_model, 'best_svm_model.m')  # save model
    y_pred_svm = best_svm_model.predict_proba(te_feat_scaled_sel_pca)
    print("SVM's logloss: ", log_loss(y_test, y_pred_svm))


if __name__ == '__main__':
    run_main()
