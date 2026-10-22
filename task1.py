from abc import ABC, abstractmethod
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class Config:
    mlflow_uri: str = "file:./mlruns"
    data_path: str = "creditcard.csv"
    test_size: float = 0.2
    cv_folds: int = 3
    random_state: int = 42

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def fit_transform(self, X, y=None):
        numerical_features = ['Amount', 'Hour']
        numerical_transformer = Pipeline([('scaler', StandardScaler())])
        self.preprocessor = ColumnTransformer(
            transformers=[('numerical', numerical_transformer, numerical_features)],
            remainder='passthrough'
        )
        X_processed = self.preprocessor.fit_transform(X)
        self.feature_names = numerical_features + [col for col in X.columns if col not in numerical_features]
        return pd.DataFrame(X_processed, columns=self.feature_names)

    def transform(self, X):
        X_processed = self.preprocessor.transform(X)
        return pd.DataFrame(X_processed, columns=self.feature_names)

class Model(ABC):
    def __init__(self, name, preprocessor, param_grid, scale_pos_weight=None):
        self.name = name
        self.preprocessor = preprocessor
        self.param_grid = param_grid
        self.scale_pos_weight = scale_pos_weight
        self.best_pipe = None
        self.best_params = None
        self.cv_score = None

    @abstractmethod
    def _create_pipeline(self):
        pass

    def train(self, X_train, y_train, cv):
        pipe = self._create_pipeline()
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=self.param_grid,
            cv=cv,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)
        self.best_pipe = gs.best_estimator_
        self.best_params = gs.best_params_
        self.cv_score = gs.best_score_
        return self.best_pipe

    def predict(self, X):
        return self.best_pipe.predict(X)

    def predict_proba(self, X):
        return self.best_pipe.predict_proba(X)

    def evaluate(self, X_test, y_test):
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

class RandomForestModel(Model):
    def _create_pipeline(self):
        return Pipeline([
            ('prep', self.preprocessor.preprocessor),
            ('clf', RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

class XGBoostModel(Model):
    def _create_pipeline(self):
        return Pipeline([
            ('prep', self.preprocessor.preprocessor),
            ('clf', XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=self.scale_pos_weight
            ))
        ])

def main():
    config = Config()
    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment("Creditcard_Model_Tuning")

    df = pd.read_csv(config.data_path)
    df.drop_duplicates(inplace=True)
    df['Hour'] = (df['Time'] // 3600) % 24
    df.drop('Time', axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size,
        random_state=config.random_state, stratify=y
    )

    preprocessor = DataPreprocessor()

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    models = [
        RandomForestModel(
            name='Random Forest',
            preprocessor=preprocessor,
            param_grid={
                'clf__n_estimators': [200],
                'clf__max_depth': [5, 8, 10],
                'clf__min_samples_leaf': [5, 10]
            }
        ),
        XGBoostModel(
            name='XGBoost',
            preprocessor=preprocessor,
            param_grid={
                'clf__n_estimators': [200],
                'clf__max_depth': [3, 4, 5],
                'clf__learning_rate': [0.05]
            },
            scale_pos_weight=scale_pos_weight
        )
    ]

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

    for model in models:
        with mlflow.start_run(run_name=f"{model.name}_Training") as run:
            run_id = run.info.run_id

            mlflow.log_param("model_name", model.name)
            mlflow.log_param("test_size", config.test_size)
            mlflow.log_param("cv_folds", config.cv_folds)

            model.train(X_train, y_train, cv)
            mlflow.log_params(model.best_params)
            mlflow.log_metric("cv_auprc", model.cv_score)

            metrics = model.evaluate(X_test, y_test)
            mlflow.log_metric("test_roc_auc", metrics['roc_auc'])
            mlflow.log_metric("test_auprc", metrics['pr_auc'])

            print(f"\n{model.name} лучший AUPRC (CV): {model.cv_score:.4f}")
            print(f"Лучшие параметры: {model.best_params}")
            print(f"ROC-AUC на тесте: {metrics['roc_auc']:.4f}")
            print(f"AUPRC на тесте: {metrics['pr_auc']:.4f}")

            mlflow.sklearn.log_model(
                model.best_pipe,
                name="best_model",
                input_example=X_train.iloc[:1]
            )

if __name__ == '__main__':
    main()