from code import Code
from analysis import analysis
from sklearn.model_selection import train_test_split
from models import Models

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df=Code.load_data("BostonHousing.csv")
    analysis.hist(df['medv'])
    features = ['rm', 'lstat', 'ptratio']
    for feature in features:
        analysis.plot_relationship(df, feature)

    target_column = 'medv'
    X = df[features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    model = Models(X_train, X_test, y_train, y_test)
    lr_res = model.linear_regression_model()
    dt_res = model.tree_model()

    print("\nСРАВНЕНИЕ:")
    print(f"{'Метрика':<8} {'Linear':<12} {'Tree':<12}")
    print(f"{'MAE':<8} {lr_res['MAE']:<12.4f} {dt_res['MAE']:<12.4f}")
    print(f"{'R²':<8} {lr_res['R2']:<12.4f} {dt_res['R2']:<12.4f}")

    best_pred = model.y_lr_pred if lr_res['R2'] > dt_res['R2'] else model.y_dt_pred
    best_name = "Linear Regression" if lr_res['R2'] > dt_res['R2'] else "Decision Tree"
    analysis_obj = analysis()
    analysis_obj.graf(y_test, best_pred, best_name)

