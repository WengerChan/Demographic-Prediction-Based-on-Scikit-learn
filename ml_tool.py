# get_best_model
from sklearn.model_selection import GridSearchCV


def get_best_model(model, X_train, y_train, params, cv=5):
    """
        交叉验证获取最优模型
        设置默认5折交叉验证
    """
    clf = GridSearchCV(model, params, cv=cv)
    clf.fit(X_train, y_train)  # 运行网格搜索
    # print(clf.grid_scores_)  # 给出不同参数情况下的评价结果
    # print(clf.best_params_)  # 描述了已取得最佳结果的参数的组合
    # print(clf.best_score_)  # 成员提供优化过程期间观察到的最好的评分
    return clf.best_estimator_  # 最优估计
