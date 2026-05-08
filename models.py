from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class Models:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def tree_model(self, feature_names=None):
        print("DecisionTreeRegressor модель")

        # Обучаем модель дерева решений
        dt_regressor = DecisionTreeRegressor(
            max_depth=3,  # глубина дерева
            min_samples_split=15,  # мин. образцов для разделения
            min_samples_leaf=8,  # мин. образцов в листе
            random_state=42
        )

        dt_regressor.fit(self.X_train, self.y_train)

        y_train_pred_dt = dt_regressor.predict(self.X_train)
        self.y_dt_pred = dt_regressor.predict(self.X_test)

        # --- ОБУЧАЮЩАЯ ВЫБОРКА ---
        mae_train = mean_absolute_error(self.y_train, y_train_pred_dt)
        mse_train = mean_squared_error(self.y_train, y_train_pred_dt)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(self.y_train, y_train_pred_dt)

        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  MAE:  {mae_train:.4f}")
        print(f"  MSE:  {mse_train:.4f}")
        print(f"  RMSE: {rmse_train:.4f}")
        print(f"  R²:   {r2_train:.4f}")

        # --- ТЕСТОВАЯ ВЫБОРКА ---
        mae_test = mean_absolute_error(self.y_test, self.y_dt_pred)  # ← используем y_dt_pred
        mse_test = mean_squared_error(self.y_test, self.y_dt_pred)  # ← используем y_dt_pred
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(self.y_test, self.y_dt_pred)

        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  MAE:  {mae_test:.4f}")
        print(f"  MSE:  {mse_test:.4f}")
        print(f"  RMSE: {rmse_test:.4f}")
        print(f"  R²:   {r2_test:.4f}")


        print("ВАЖНОСТЬ ПРИЗНАКОВ (feature_importances_):")

        if feature_names is None:
            feature_names = [f"Признак_{i + 1}" for i in range(len(dt_regressor.feature_importances_))]

        for name, importance in zip(feature_names, dt_regressor.feature_importances_):
            print(f"  {name}: {importance:.4f}")

        print("\nИНТЕРПРЕТАЦИЯ ДЛЯ БИЗНЕСА:")
        print("  Важность признака показывает, насколько он влияет на прогноз")
        print("  (сумма всех важностей = 1.0)")

        # Находим самый важный признак
        max_idx = np.argmax(dt_regressor.feature_importances_)
        print(f"\n  → Самый важный признак: {feature_names[max_idx]} "
              f"(важность = {dt_regressor.feature_importances_[max_idx]:.4f})")
        return {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}


    def linear_regression_model(self, feature_names=None):
        """
        Линейная регрессия для решения задач регрессии.
        Метрики: MSE, RMSE, R².
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        print("LinearRegression модель")

        model = LinearRegression()
        model.fit(X_train_scaled, self.y_train)

        y_train_pred_lr = model.predict(X_train_scaled)
        self.y_lr_pred = model.predict(X_test_scaled)

        # --- ОБУЧАЮЩАЯ ВЫБОРКА ---
        mae_train = mean_absolute_error(self.y_train, y_train_pred_lr)
        mse_train = mean_squared_error(self.y_train, y_train_pred_lr)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(self.y_train, y_train_pred_lr)

        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  MAE:  {mae_train:.4f}")
        print(f"  MSE:  {mse_train:.4f}")
        print(f"  RMSE: {rmse_train:.4f}")
        print(f"  R²:   {r2_train:.4f}")

        # --- ТЕСТОВАЯ ВЫБОРКА ---
        mae_test = mean_absolute_error(self.y_test, self.y_lr_pred)
        mse_test = mean_squared_error(self.y_test, self.y_lr_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(self.y_test, self.y_lr_pred)

        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  MAE:  {mae_test:.4f}")
        print(f"  MSE:  {mse_test:.4f}")
        print(f"  RMSE: {rmse_test:.4f}")
        print(f"  R²:   {r2_test:.4f}")

        print("КОЭФФИЦИЕНТЫ МОДЕЛИ:")

        print(f"\nСвободный член (intercept): {model.intercept_:.4f}")

        if feature_names is None:
            feature_names = [f"Признак_{i + 1}" for i in range(len(model.coef_))]

        print("\nКоэффициенты признаков (coef_):")
        for name, coef in zip(feature_names, model.coef_):
            print(f"  {name}: {coef:.4f}")

        print("ИНТЕРПРЕТАЦИЯ ДЛЯ БИЗНЕСА:")

        for name, coef in zip(feature_names, model.coef_):
            if coef > 0:
                print(f"  • {name}: +{coef:.4f} → при увеличении {name} на 1, прогноз растёт на {coef:.4f}")
            else:
                print(f"  • {name}: {coef:.4f} → при увеличении {name} на 1, прогноз падает на {abs(coef):.4f}")

        # Конкретный пример (замените 'RM' на ваш признак, если нужно)
        if 'RM' in feature_names:
            idx = list(feature_names).index('RM')
            rm_coef = model.coef_[idx]
            print(f"\n  → Пример: рост комнатности (RM) на 1 увеличивает прогноз на {rm_coef:.4f} единиц")

        return {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}