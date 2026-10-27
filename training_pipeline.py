import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
from model_builder import ExtractableAutoencoder, CrossValidationTracker, create_model_pipeline
from matplotlib import rcParams

# Validation R² is computed as the mean of the output-wise R² scores.
plt.rcParams['axes.unicode_minus'] = False

class CrossValidationPipeline:
    """Five-fold cross-validation training pipeline
    Run a single five-fold cross-validation workflow"""

    def __init__(self, results_dir='cv_results'):
        self.results_dir = results_dir
        self.cv_history = {}
        self.best_models = {}
        self.scalers = {}

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def create_scaler_pipeline(self, X_train, y_train):
        """Create scalers for the current training fold"""
        # Feature scalers
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # Fit the scalers
        X_scaled = scaler_x.fit_transform(X_train)
        y_scaled = scaler_y.fit_transform(y_train)

        return scaler_x, scaler_y, X_scaled, y_scaled

    def train_fold(self, fold_data, X_train, y_train, params,
                   epochs=200, fold_num=1):
        """Train one cross-validation fold
        params: one hyperparameter configuration
        fold_num: current fold number
        """
        print(f"\n=== Training fold {fold_num} ===")

        # Get the training and validation indices for this fold
        train_idx = fold_data['train_indices']
        val_idx = fold_data['val_indices']

        # Select data for the current fold
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Fit scalers on the current fold training subset to prevent data leakage
        scaler_x, scaler_y, X_train_scaled, y_train_scaled = self.create_scaler_pipeline(
            X_fold_train, y_fold_train)

        X_val_scaled = scaler_x.transform(X_fold_val)
        y_val_scaled = scaler_y.transform(y_fold_val)

        # Save the scalers
        self.scalers[fold_num] = {'x': scaler_x, 'y': scaler_y}

        # Build the model
        model_builder, model = create_model_pipeline(
            input_dim=3,
            encoding_dim=1,
            hidden_layers=params.get('hidden_layers', [128, 64]),
            activation=params.get('activation', 'relu'),
            dropout_rate=params.get('dropout_rate', 0.1),
            learning_rate=params.get('learning_rate', 0.001)
        )

        # Create a callback to record training history
        cv_tracker = CrossValidationTracker(fold_num)

        # Train the model
        history = model.fit(
            [X_train_scaled, y_train_scaled],          #Model inputs
            X_train_scaled,                            #Reconstruction targets
            batch_size=params.get('batch_size', 16),
            epochs=epochs,
            validation_data=([X_val_scaled, y_val_scaled], X_val_scaled),
            verbose=1,
            callbacks=[cv_tracker]                     #Record per-epoch metrics such as loss and MAE
        )

        # Record training history
        self.cv_history[fold_num] = {
            'tracker': cv_tracker,
            'keras_history': history.history
        }

        # Evaluate the current fold
        val_pred = model.predict([X_val_scaled, y_val_scaled], verbose=0)
        val_r2 = r2_score(X_val_scaled, val_pred)

        print(f"Fold {fold_num} validation R²: {val_r2:.4f}")
        #model_builder: Model builder used to access components such as the decoder
        #model: Compiled full model
        #val_r2: Validation R²
        return model_builder, model, val_r2

    def run_cross_validation(self, X_train, y_train, cv_data,
                             params, epochs=200):
        """Run five-fold cross-validation; cv_data contains the fold indices"""
        print("=" * 60)
        print("Starting five-fold cross-validation")
        print("=" * 60)
        print(f"Parameters: {params}")
        print(f"Epochs: {epochs}")

        fold_scores = []

        for fold_data in cv_data:
            fold_num = fold_data['fold']

            model_builder, model, val_r2 = self.train_fold(
                fold_data, X_train, y_train, params, epochs, fold_num)

            fold_scores.append(val_r2)

            # Store the model for the current fold
            self.best_models[fold_num] = {
                'model_builder': model_builder,
                'model': model,
                'score': val_r2
            }

            # Save model components for this fold
            # Create a directory for this fold
            model_dir = os.path.join(self.results_dir, f'fold_{fold_num}')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            #Save the encoder, decoder, and full autoencoder as H5 files
            model_builder.save_components(os.path.join(model_dir, 'autoencoder'))
            #Save the scalers
            joblib.dump(self.scalers[fold_num],
                        os.path.join(model_dir, 'scaler.pkl'))

        # Calculate the mean score across folds
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(f"\n=== Cross-validation results ===")
        print(f"Fold R² scores: {fold_scores}")
        print(f"Mean R²: {avg_score:.4f} ± {std_score:.4f}")

        return avg_score, fold_scores

    def plot_training_history(self, save_path=None):
        """Plot training history; save_path is the output file path"""
        n_folds = len(self.cv_history)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics = ['loss', 'val_loss', 'mae', 'val_mae']
        titles = ['Training loss', 'Validation loss', 'Training MAE', 'Validation MAE']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            for fold_num, history_data in self.cv_history.items():
                # Use the history recorded by the callback
                if metric in history_data['tracker'].history:
                    data = history_data['tracker'].history[metric]
                    axes[i].plot(data, label=f'Fold {fold_num}')

            axes[i].set_title(title)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training-history plot saved to: {save_path}")

        plt.show()

    def save_results(self, params, avg_score, fold_scores):
        """Save cross-validation results
        Save fold scores, mean R², and hyperparameters in a table
        params contains one hyperparameter configuration
        """
        n_folds = len(fold_scores)

        # Ensure all columns have matching lengths
        data = {
            'fold': list(range(1, n_folds + 1)),
            'r2_score': fold_scores,
            'avg_r2': [avg_score] * n_folds
        }

        # Format parameter values to match the number of folds
        for key, value in params.items():
            # Convert parameter values to lists of matching length
            if isinstance(value, (list, np.ndarray)) and len(value) == n_folds:
                data[key] = value
            else:
                data[key] = [value] * n_folds

        results_df = pd.DataFrame(data)
        #Save results to cv_results.csv in results_dir
        results_path = os.path.join(self.results_dir, 'cv_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"See the output above for details.")

        # Save training history
        history_data = {}
        for fold_num, history in self.cv_history.items():
            history_data[f'fold_{fold_num}'] = history['tracker'].history

        history_path = os.path.join(self.results_dir, 'training_history.pkl')
        joblib.dump(history_data, history_path)
        print(f"Training history saved to: {history_path}")


class GridSearchCVPipeline:
    """Grid-search cross-validation pipeline
    Run grid search over hyperparameter combinations using cross-validation"""


    def __init__(self, results_dir='grid_search_cv_results'):
        # Create the grid-search results directory
        self.results_dir = results_dir
        self.search_results = []

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def search(self, X_train, y_train, cv_data, param_grid, epochs=200):
        """Run grid-search cross-validation"""
        print("=" * 60)
        print("Run grid search cross-validation.")
        print("=" * 60)

        param_combinations = list(ParameterGrid(param_grid))
        print(f"Total hyperparameter combinations: {len(param_combinations)}")

        best_score = -float('inf')
        best_params = None

        for i, params in enumerate(param_combinations, 1):
            print(f"\n--- Combination {i}/{len(param_combinations)} ---")
            print(f"Parameters: {params}")

            # Create a separate results directory for each parameter combination
            cv_pipeline = CrossValidationPipeline(
                results_dir=os.path.join(self.results_dir, f'combination_{i}')
            )

            # Run cross-validation
            avg_score, fold_scores = cv_pipeline.run_cross_validation(
                X_train, y_train, cv_data, params, epochs)

            # Plot training history
            cv_pipeline.plot_training_history(
                os.path.join(cv_pipeline.results_dir, 'training_history.png'))

            # Save results
            cv_pipeline.save_results(params, avg_score, fold_scores)

            # Record search results
            result = params.copy()
            result.update({
                'avg_r2': avg_score,
                'fold_scores': fold_scores,
                'combination_id': i
            })
            self.search_results.append(result)

            print(f"Mean R²: {avg_score:.4f}")

            # Update the best parameters
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                print("New best hyperparameter configuration found.")

        # Save grid-search results
        self.save_search_results()

        print(f"\n=== Grid search completed ===")
        print(f"Best parameters: {best_params}")
        print(f"Best mean R²: {best_score:.4f}")

        return best_params, best_score

    def save_search_results(self):
        """Save grid-search results"""
        if not self.search_results:
            return

        results_df = pd.DataFrame(self.search_results)
        results_path = os.path.join(self.results_dir, 'grid_search_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Grid-search results saved to: {results_path}")


# Example usage
if __name__ == "__main__":
    from preprocess_data import load_preprocessed_data

    # Load the preprocessed data
    data_dict = load_preprocessed_data('processed_data_cv')
    if data_dict is None:
        print("Failed to load preprocessed data.")
        exit()

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    cv_data = data_dict['cv_data']

    print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Number of CV folds: {len(cv_data)}")

    # Define the hyperparameter grid
    param_grid = {
        'learning_rate': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
        'dropout_rate': [0.0, 0.05, 0.1, 0.15, 0.2],
        'batch_size': [8, 16, 24, 32],
        'activation': ['relu', 'tanh'],
        'hidden_layers': [[128, 64]]
    }

    # Run grid-search cross-validation
    grid_search = GridSearchCVPipeline('grid_search_cv_results')
    best_params, best_score = grid_search.search(
        X_train, y_train, cv_data, param_grid, epochs=200)