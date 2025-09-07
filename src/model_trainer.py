import numpy as np
import pickle
import config.config as cfg
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionModel:
    def __init__(self):
        self.model = None
        self.feature_importances = None
        self.X_train = None
        self.y_train = None

    def build_model(self, model_type: str = 'svm'):
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.3,
                random_state=cfg.RANDOM_SEED,
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_type == 'svm':
            model = SVC(
                kernel='linear',
                C=0.1,
                random_state=cfg.RANDOM_SEED,
                class_weight='balanced',
                probability=True
            )
        elif model_type == 'linear_svc':
            model = LinearSVC(
                C=0.1,
                random_state=cfg.RANDOM_SEED,
                class_weight='balanced',
                dual=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type in ['svm', 'linear_svc']:
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
        else:
            self.model = model

        return self.model

    def train(self, X_train, y_train, model_type='svm'):
        self.X_train = X_train
        self.y_train = y_train

        logger.info(f"Training {model_type.upper()} on {len(X_train)} samples...")
        logger.info(f"Class distribution: {np.bincount(y_train)}")

        self.build_model(model_type=model_type)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_SEED)
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                    cv=cv, scoring='f1_weighted', n_jobs=-1)

        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')

        logger.info(f"Training accuracy: {train_acc:.4f}")
        logger.info(f"Training F1 score: {train_f1:.4f}")

        return train_acc, train_f1

    def evaluate(self, X_test, y_test, save_plots=True):
        y_pred = self.model.predict(X_test)

        try:
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)
            elif hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(X_test)
                if decision_scores.ndim == 1:
                    y_prob = 1 / (1 + np.exp(-decision_scores))
                    y_prob = np.vstack([1 - y_prob, y_prob]).T
                else:
                    y_prob = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
            else:
                y_prob = np.ones((len(y_pred), len(np.unique(y_test)))) / len(np.unique(y_test))
        except Exception as e:
            logger.warning(f"Could not compute probabilities: {e}")
            y_prob = np.ones((len(y_pred), len(np.unique(y_test)))) / len(np.unique(y_test))

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        report = classification_report(
            y_test, y_pred,
            target_names=list(cfg.CATEGORIES.keys()),
            output_dict=True
        )

        cm = confusion_matrix(y_test, y_pred)

        if save_plots:
            self._plot_confusion_matrix(cm, list(cfg.CATEGORIES.keys()))

        class_metrics = {}
        for i, class_name in enumerate(cfg.CATEGORIES.keys()):
            class_metrics[class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1_score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            }

        return {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'f1_macro': f1_macro,
            'predictions': y_pred,
            'probabilities': y_prob,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_metrics': class_metrics
        }

    def _plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(str(cfg.MODEL_DIR / 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Confusion matrix saved")

    def save_model(self, path: str = None):
        if path is None:
            path = cfg.MODEL_DIR / 'anomaly_detection_model.joblib'
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str = None):
        if path is None:
            path = cfg.MODEL_DIR / 'anomaly_detection_model.joblib'
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(str(path))
        logger.info(f"Model loaded from {path}")

    def predict(self, features: np.ndarray):
        if features.ndim == 1:
            features = features.reshape(1, -1)
        prediction = self.model.predict(features)
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(features)
        else:
            decision_function = self.model.decision_function(features)
            if decision_function.ndim == 1:
                probability = np.vstack([1 - decision_function, decision_function]).T
            else:
                probability = np.exp(decision_function) / np.sum(np.exp(decision_function), axis=1, keepdims=True)
        return prediction[0], probability[0]

    def analyze_performance(self, X_test, y_test):
        results = self.evaluate(X_test, y_test, save_plots=False)
        logger.info("\n" + "=" * 50)
        logger.info("MODEL PERFORMANCE ANALYSIS")
        logger.info("=" * 50)
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Weighted F1 Score: {results['f1_weighted']:.4f}")
        logger.info(f"Macro F1 Score: {results['f1_macro']:.4f}")

        logger.info("\nPer-class Performance:")
        for class_name, metrics in results['class_metrics'].items():
            logger.info(f"{class_name:15s}: Precision={metrics['precision']:.3f}, "
                        f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, "
                        f"Support={metrics['support']}")

        if self.X_train is not None:
            train_pred = self.model.predict(self.X_train)
            train_acc = accuracy_score(self.y_train, train_pred)
            overfitting_gap = train_acc - results['accuracy']
            if overfitting_gap > 0.15:
                logger.warning(f"⚠️ Potential overfitting detected! (Train-Test gap: {overfitting_gap:.3f})")
            else:
                logger.info(f"✓ Good generalization (Train-Test gap: {overfitting_gap:.3f})")

        return results
