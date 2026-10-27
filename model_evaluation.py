import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
from preprocess_data import load_preprocessed_data
from model_builder import create_model_pipeline
from datetime import datetime

# Configure plotting to display minus signs correctly.
plt.rcParams['axes.unicode_minus'] = False


class FinalModelTrainer:
    """Train the final model on the full training set using selected hyperparameters"""

    def __init__(self, results_dir='final_model_results_03_500'):
        self.results_dir = results_dir
        self.final_scaler = None
        self.final_model = None
        self.model_builder = None

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def train_final_model(self, X_train, y_train, best_params, epochs=200):
        """Train the final model on the full training set"""
        print("=" * 60)
        print("Training the final model on the full training set")
        print("=" * 60)

        # 1. Fitting scalers on the full training set (100 samples)
        print("Fitting scalers on the 100 training samples...")
        self.final_scaler = {}
        self.final_scaler['x'] = StandardScaler()
        self.final_scaler['y'] = StandardScaler()

        # Save the output artifact.
        X_train_scaled = self.final_scaler['x'].fit_transform(X_train)
        y_train_scaled = self.final_scaler['y'].fit_transform(y_train)

        print("Scalers fitted successfully")

        # 2. Build and train the final model using the selected hyperparameters
        print("Building the final model...")
        self.model_builder, self.final_model = create_model_pipeline(
            input_dim=3,
            encoding_dim=1,
            hidden_layers=best_params.get('hidden_layers', [128, 64]),
            activation=best_params.get('activation', 'relu'),
            dropout_rate=best_params.get('dropout_rate', 0.15),
            learning_rate=best_params.get('learning_rate', 0.002)
        )

        # Save the output artifact.
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=best_params.get('learning_rate', 0.002))
        self.final_model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        print("Final model created")
        print(f"Hyperparameters: {best_params}")

        # 3. Training the final model
        print("Training the final model...")
        history = self.final_model.fit(
            [X_train_scaled, y_train_scaled],
            X_train_scaled,
            batch_size=best_params.get('batch_size', 8),
            epochs=epochs,
            verbose=1,
            validation_split=0.0  # Train on the full training set without a validation split
        )

        # 4. Predict and evaluate on the training set
        train_predictions_scaled = self.final_model.predict(
            [X_train_scaled, y_train_scaled], verbose=0)
        train_predictions = self.final_scaler['x'].inverse_transform(
            train_predictions_scaled)

        # Calculate training-set metrics
        train_mse = mean_squared_error(X_train, train_predictions)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(X_train, train_predictions)

        # Calculate R² for each output feature
        train_r2_per_feature = []
        for i in range(3):
            r2_feature = r2_score(X_train[:, i], train_predictions[:, i])
            train_r2_per_feature.append(r2_feature)

        print("\n=== Training-set evaluation results ===")
        print(f"Training MSE: {train_mse:.6f}")
        print(f"Training RMSE: {train_rmse:.6f}")
        print(f"Training overall R²: {train_r2:.6f}")
        for i, r2 in enumerate(train_r2_per_feature):
            print(f"Training feature {i + 1} R²: {r2:.6f}")

        # 5. Save training predictions in the original scale
        train_pred_file = self._save_train_predictions(X_train, train_predictions, y_train)

        # 6. Save the final model and scalers
        self._save_final_components()

        # Save the output artifact.
        train_results = {
            'predictions': train_predictions,
            'mse': train_mse,
            'rmse': train_rmse,
            'r2': train_r2,
            'r2_per_feature': train_r2_per_feature,
            'X_train_original': X_train,
            'y_train_original': y_train,
            'prediction_file': train_pred_file
        }
        self._save_train_results(train_results)

        print("Final model trained and saved")
        return history, train_results

    def _save_train_predictions(self, X_train_original, train_predictions, y_train_original):
        """Save training predictions in the original scale"""
        try:
            # Create the prediction output directory
            pred_dir = os.path.join(self.results_dir, 'train_predictions')
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create a detailed prediction DataFrame
            n_samples = X_train_original.shape[0]
            feature_names = ['Shear_rate(/s)', 'Temperature(℃)', 'Concentration(%)']

            # Create a DataFrame containing all output fields
            predictions_df = pd.DataFrame()

            # Add sample indices
            predictions_df['sample_index'] = range(n_samples)
            predictions_df['dataset_type'] = 'training_set'

            # Add actual and predicted values for each feature
            for i, feature_name in enumerate(feature_names):
                predictions_df[f'{feature_name}_actual'] = X_train_original[:, i]
                predictions_df[f'{feature_name}_predicted'] = train_predictions[:, i]
                predictions_df[f'{feature_name}_error'] = (
                        X_train_original[:, i] - train_predictions[:, i]
                )
                predictions_df[f'{feature_name}_abs_error'] = np.abs(
                    X_train_original[:, i] - train_predictions[:, i]
                )
                predictions_df[f'{feature_name}_relative_error'] = (
                        np.abs(X_train_original[:, i] - train_predictions[:, i]) /
                        (np.abs(X_train_original[:, i]) + 1e-8) * 100  # Avoid division by zero and convert to a percentage
                )

            # Add the actual shear-stress labels
            predictions_df['Shear_stress(Pas)_actual'] = y_train_original.flatten()

            # Add summary evaluation metrics
            predictions_df['mse_overall'] = mean_squared_error(
                X_train_original.flatten(), train_predictions.flatten()
            )
            predictions_df['rmse_overall'] = np.sqrt(predictions_df['mse_overall'].iloc[0])
            predictions_df['r2_overall'] = r2_score(
                X_train_original, train_predictions
            )
            predictions_df['mae_overall'] = mean_absolute_error(
                X_train_original.flatten(), train_predictions.flatten()
            )

            # Add evaluation timestamp and model information
            predictions_df['evaluation_timestamp'] = timestamp
            predictions_df['total_samples'] = n_samples

            # Save to CSV
            train_pred_file = os.path.join(pred_dir, f'train_predictions_{timestamp}.csv')
            predictions_df.to_csv(train_pred_file, index=False)

            # Also save an Excel copy for convenient inspection
            train_pred_excel_file = os.path.join(pred_dir, f'train_predictions_{timestamp}.xlsx')
            predictions_df.to_excel(train_pred_excel_file, index=False)

            print(f"Training predictions saved:")
            print(f"  CSV: {train_pred_file}")
            print(f"  Excel: {train_pred_excel_file}")
            print(f"  Number of samples: {n_samples}")
            print(f"  Number of features: {len(feature_names)}")

            return train_pred_file

        except Exception as e:
            print(f"Error saving training predictions: {e}")
            return None

    def _save_final_components(self):
        """Save the final model components and scalers"""
        # Save model components
        self.model_builder.save_components(
            os.path.join(self.results_dir, 'final_autoencoder'))

        # Save the output artifact.
        joblib.dump(self.final_scaler,
                    os.path.join(self.results_dir, 'final_scaler.pkl'))

        print(f"Final model components saved to: {self.results_dir}")

    def _save_train_results(self, train_results):
        """Save training-set evaluation metrics."""
        results_df = pd.DataFrame([{
            'mse': train_results['mse'],
            'rmse': train_results['rmse'],
            'r2_overall': train_results['r2'],
            'r2_feature1': train_results['r2_per_feature'][0],
            'r2_feature2': train_results['r2_per_feature'][1],
            'r2_feature3': train_results['r2_per_feature'][2],
            'prediction_file': train_results.get('prediction_file', '')
        }])

        results_path = os.path.join(self.results_dir, 'final_model_train_evaluation.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Training evaluation results saved to: {results_path}")

    def load_final_model(self):
        """Load the final model"""
        try:
            # Load model components
            encoder = tf.keras.models.load_model(
                os.path.join(self.results_dir, 'final_autoencoder_encoder.h5'))
            decoder = tf.keras.models.load_model(
                os.path.join(self.results_dir, 'final_autoencoder_decoder.h5'))
            autoencoder = tf.keras.models.load_model(
                os.path.join(self.results_dir, 'final_autoencoder_autoencoder.h5'))

            # Save the output artifact.
            scaler = joblib.load(os.path.join(self.results_dir, 'final_scaler.pkl'))

            self.final_scaler = scaler
            self.final_model = autoencoder

            # Recreate the model builder if needed
            from model_builder import ExtractableAutoencoder
            self.model_builder = ExtractableAutoencoder()
            self.model_builder.encoder = encoder
            self.model_builder.decoder = decoder
            self.model_builder.autoencoder = autoencoder

            print("Final model loaded successfully.")
            return True

        except Exception as e:
            print(f"Error loading the final model: {e}")
            return False


class CVEvaluation:
    """Evaluate the final model on the independent test set"""

    def __init__(self, results_dir='final_model_results_03_500'):
        self.results_dir = results_dir
        self.final_trainer = FinalModelTrainer(results_dir)

    def evaluate_on_testset(self, X_test, y_test):
        """Evaluate the final model on the test set"""
        print("\n=== Test-set evaluation using the final model and fitted scalers===")

        if self.final_trainer.final_model is None:
            if not self.final_trainer.load_final_model():
                print("Unable to load the final model.")
                return None

        # Scale the test set using the training-set scalers
        X_test_scaled = self.final_trainer.final_scaler['x'].transform(X_test)
        y_test_scaled = self.final_trainer.final_scaler['y'].transform(y_test)

        # Generate predictions
        y_pred_scaled = self.final_trainer.final_model.predict(
            [X_test_scaled, y_test_scaled], verbose=0)

        # Inverse-transform predictions using the training-set scaler
        y_pred_original = self.final_trainer.final_scaler['x'].inverse_transform(
            y_pred_scaled)

        # Calculate evaluation metrics
        mse = mean_squared_error(X_test, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(X_test, y_pred_original)

        # Calculate R² for each output feature
        r2_per_feature = []
        for i in range(3):
            r2_feature = r2_score(X_test[:, i], y_pred_original[:, i])
            r2_per_feature.append(r2_feature)

        print("Test-set evaluation results:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Overall R²: {r2:.6f}")
        for i, r2_val in enumerate(r2_per_feature):
            print(f"Feature {i + 1} R²: {r2_val:.6f}")

        # Save the output artifact.
        test_pred_file = self._save_test_predictions(X_test, y_pred_original, y_test)

        results = {
            'predictions': y_pred_original,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'r2_per_feature': r2_per_feature,
            'X_test_original': X_test,
            'y_test_original': y_test,
            'prediction_file': test_pred_file
        }

        # Save evaluation results
        self._save_evaluation_results(results)

        return results

    def _save_test_predictions(self, X_test_original, test_predictions, y_test_original):
        """Save test-set predictions in the original scale."""
        try:
            # Create the prediction output directory
            pred_dir = os.path.join(self.results_dir, 'test_predictions')
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create a detailed prediction DataFrame
            n_samples = X_test_original.shape[0]
            feature_names = ['Shear_rate(/s)', 'Temperature(℃)', 'Concentration(%)']

            # Create a DataFrame containing all output fields
            predictions_df = pd.DataFrame()

            # Add sample indices
            predictions_df['sample_index'] = range(n_samples)
            predictions_df['is_test_set'] = True
            predictions_df['dataset_type'] = 'test_set'

            # Add actual and predicted values for each feature
            for i, feature_name in enumerate(feature_names):
                predictions_df[f'{feature_name}_actual'] = X_test_original[:, i]
                predictions_df[f'{feature_name}_predicted'] = test_predictions[:, i]
                predictions_df[f'{feature_name}_error'] = (
                        X_test_original[:, i] - test_predictions[:, i]
                )
                predictions_df[f'{feature_name}_abs_error'] = np.abs(
                    X_test_original[:, i] - test_predictions[:, i]
                )
                predictions_df[f'{feature_name}_relative_error'] = (
                        np.abs(X_test_original[:, i] - test_predictions[:, i]) /
                        (np.abs(X_test_original[:, i]) + 1e-8) * 100  # Avoid division by zero and convert to a percentage
                )

            # Add the actual shear-stress labels
            predictions_df['Shear_stress(Pas)_actual'] = y_test_original.flatten()

            # Add summary evaluation metrics
            predictions_df['mse_overall'] = mean_squared_error(
                X_test_original.flatten(), test_predictions.flatten()
            )
            predictions_df['rmse_overall'] = np.sqrt(predictions_df['mse_overall'].iloc[0])
            predictions_df['r2_overall'] = r2_score(
                X_test_original, test_predictions
            )
            predictions_df['mae_overall'] = mean_absolute_error(
                X_test_original.flatten(), test_predictions.flatten()
            )

            # Add evaluation timestamp and model information
            predictions_df['evaluation_timestamp'] = timestamp
            predictions_df['total_samples'] = n_samples

            # Save to CSV
            test_pred_file = os.path.join(pred_dir, f'test_predictions_{timestamp}.csv')
            predictions_df.to_csv(test_pred_file, index=False)

            # Also save an Excel copy for convenient inspection
            test_pred_excel_file = os.path.join(pred_dir, f'test_predictions_{timestamp}.xlsx')
            predictions_df.to_excel(test_pred_excel_file, index=False)

            print("Test predictions saved:")
            print(f"  CSV: {test_pred_file}")
            print(f"  Excel: {test_pred_excel_file}")
            print(f"  Number of samples: {n_samples}")
            print(f"  Number of features: {len(feature_names)}")

            return test_pred_file

        except Exception as e:
            print(f"Error saving output: {e}")
            return None

    def _save_evaluation_results(self, results):
        """Save evaluation results"""
        results_df = pd.DataFrame([{
            'mse': results['mse'],
            'rmse': results['rmse'],
            'r2_overall': results['r2'],
            'r2_feature1': results['r2_per_feature'][0],
            'r2_feature2': results['r2_per_feature'][1],
            'r2_feature3': results['r2_per_feature'][2],
            'prediction_file': results.get('prediction_file', '')
        }])

        results_path = os.path.join(self.results_dir, 'final_model_test_evaluation.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Test evaluation results saved to: {results_path}")

    def save_combined_predictions(self, train_results, test_results):
        """Combine and save predictions for the training and test sets."""
        try:
            # Create the combined-output directory
            combined_dir = os.path.join(self.results_dir, 'combined_predictions')
            if not os.path.exists(combined_dir):
                os.makedirs(combined_dir)

            # Generate a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare training-set data
            train_df = pd.DataFrame()
            train_df['sample_index'] = range(len(train_results['X_train_original']))
            train_df['dataset_type'] = 'training'
            train_df['is_test_set'] = False

            # Prepare test-set data
            test_df = pd.DataFrame()
            test_df['sample_index'] = range(len(test_results['X_test_original']))
            test_df['dataset_type'] = 'test'
            test_df['is_test_set'] = True

            feature_names = ['Shear_rate(/s)', 'Temperature(℃)', 'Concentration(%)']

            # Combine feature data
            for i, feature_name in enumerate(feature_names):
                # Training set
                train_df[f'{feature_name}_actual'] = train_results['X_train_original'][:, i]
                train_df[f'{feature_name}_predicted'] = train_results['predictions'][:, i]
                train_df[f'{feature_name}_error'] = (
                        train_results['X_train_original'][:, i] - train_results['predictions'][:, i]
                )
                train_df[f'{feature_name}_abs_error'] = np.abs(
                    train_results['X_train_original'][:, i] - train_results['predictions'][:, i]
                )

                # Test set
                test_df[f'{feature_name}_actual'] = test_results['X_test_original'][:, i]
                test_df[f'{feature_name}_predicted'] = test_results['predictions'][:, i]
                test_df[f'{feature_name}_error'] = (
                        test_results['X_test_original'][:, i] - test_results['predictions'][:, i]
                )
                test_df[f'{feature_name}_abs_error'] = np.abs(
                    test_results['X_test_original'][:, i] - test_results['predictions'][:, i]
                )

            # Combine target labels
            train_df['Shear_stress(Pas)_actual'] = train_results['y_train_original'].flatten()
            test_df['Shear_stress(Pas)_actual'] = test_results['y_test_original'].flatten()

            # Combine the two datasets
            combined_df = pd.concat([train_df, test_df], ignore_index=True)

            # Add evaluation metrics
            combined_df['evaluation_timestamp'] = timestamp
            combined_df['total_samples'] = len(combined_df)

            # Calculate overall evaluation metrics
            X_combined = np.vstack([train_results['X_train_original'], test_results['X_test_original']])
            y_pred_combined = np.vstack([train_results['predictions'], test_results['predictions']])

            combined_df['mse_overall'] = mean_squared_error(X_combined.flatten(), y_pred_combined.flatten())
            combined_df['rmse_overall'] = np.sqrt(combined_df['mse_overall'].iloc[0])
            combined_df['r2_overall'] = r2_score(X_combined, y_pred_combined)
            combined_df['mae_overall'] = mean_absolute_error(X_combined.flatten(), y_pred_combined.flatten())

            # Save combined data
            combined_file = os.path.join(combined_dir, f'combined_predictions_{timestamp}.csv')
            combined_df.to_csv(combined_file, index=False)

            combined_excel_file = os.path.join(combined_dir, f'combined_predictions_{timestamp}.xlsx')
            combined_df.to_excel(combined_excel_file, index=False)

            print("Combined predictions saved:")
            print(f"  Training samples: {len(train_df)}")
            print(f"  Test samples: {len(test_df)}")
            print(f"  Total samples: {len(combined_df)}")
            print(f"  File path: {combined_file}")

            return combined_file

        except Exception as e:
            print(f"Error saving output: {e}")
            return None

    def plot_predictions_vs_actual(self, X_actual, y_pred, title="predictions vs actual",
                                   save_name="predictions_vs_actual.png", dataset_type="Training"):
        """Plot predicted versus actual values for each output feature."""
        # Set figure size and subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Set feature names and colors
        feature_names = ['Shear rate (/s)', 'Temperature (℃)', 'Concentration (%)']

        for i in range(3):
            # Calculate MAE and R² for the current feature
            mae = mean_absolute_error(X_actual[:, i], y_pred[:, i])
            r2 = r2_score(X_actual[:, i], y_pred[:, i])

            # Plot scatter points with transparency
            axes[i].scatter(X_actual[:, i], y_pred[:, i],
                            color='blue', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

            # Determine the diagonal reference-line range
            min_val = min(X_actual[:, i].min(), y_pred[:, i].min())
            max_val = max(X_actual[:, i].max(), y_pred[:, i].max())

            # Plot a red dashed reference line
            axes[i].plot([min_val, max_val], [min_val, max_val],
                         'r--', linewidth=1.5, alpha=0.7, label='y=x')

            # Set axis labels
            axes[i].set_xlabel(f'Actual {feature_names[i]}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(f'Predicted {feature_names[i]}', fontsize=12, fontweight='bold')

            # Set the title
            axes[i].set_title(f'{feature_names[i]}', fontsize=13, fontweight='bold', pad=10)

            # Add a light-gray grid
            axes[i].grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)

            # Save the output artifact.
            text_str = f'MAE = {mae:.3f}\n$R^{{2}}$ = {r2:.3f}'
            axes[i].text(0.05, 0.95, text_str, transform=axes[i].transAxes,
                         fontsize=11, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

            # Set axis limits
            buffer = 0.05 * (max_val - min_val)
            axes[i].set_xlim([min_val - buffer, max_val + buffer])
            axes[i].set_ylim([min_val - buffer, max_val + buffer])

            # Set tick locations
            axes[i].tick_params(axis='both', which='major', labelsize=10)

            # Add a legend
            axes[i].legend(loc='lower right', fontsize=9, framealpha=0.9)

        # Set the overall title
        plt.suptitle(f'{dataset_type} Prediction Results', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the output artifact.
        plot_path = os.path.join(self.results_dir, save_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Prediction comparison plot saved to: {plot_path}")

        plt.show()

    def plot_combined_predictions(self, X_train_actual, y_train_pred,
                                  X_test_actual, y_test_pred, title="Predicted vs. actual values for all data"):
        """Plot predicted versus actual values for training and test samples."""
        # Combine the training and test predictions.
        X_combined = np.vstack([X_train_actual, X_test_actual])
        y_pred_combined = np.vstack([y_train_pred, y_test_pred])

        # Save the output artifact.
        train_labels = np.zeros(len(X_train_actual))
        test_labels = np.ones(len(X_test_actual))
        combined_labels = np.hstack([train_labels, test_labels])

        # Create the figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        feature_names = ['Shear rate (/s)', 'Temperature (℃)', 'Concentration (%)']

        for i in range(3):
            # Calculate metrics for each dataset
            # Training set
            train_mask = combined_labels == 0
            train_mae = mean_absolute_error(X_combined[train_mask, i], y_pred_combined[train_mask, i])
            train_r2 = r2_score(X_combined[train_mask, i], y_pred_combined[train_mask, i])

            # Test set
            test_mask = combined_labels == 1
            test_mae = mean_absolute_error(X_combined[test_mask, i], y_pred_combined[test_mask, i])
            test_r2 = r2_score(X_combined[test_mask, i], y_pred_combined[test_mask, i])

            axes[i].scatter(X_combined[train_mask, i], y_pred_combined[train_mask, i],
                            color='blue', alpha=0.6, s=40, edgecolors='white',
                            linewidth=0.5, label='Training', marker='o')

            axes[i].scatter(X_combined[test_mask, i], y_pred_combined[test_mask, i],
                            color='red', alpha=0.6, s=40, edgecolors='white',
                            linewidth=0.5, label='Testing', marker='s')

            # Determine the diagonal reference-line range
            min_val = min(X_combined[:, i].min(), y_pred_combined[:, i].min())
            max_val = max(X_combined[:, i].max(), y_pred_combined[:, i].max())

            # Plot the diagonal reference line
            axes[i].plot([min_val, max_val], [min_val, max_val],
                         'k--', linewidth=1.5, alpha=0.7, label='y=x')

            # Set axis labels
            axes[i].set_xlabel(f'Actual {feature_names[i]}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(f'Predicted {feature_names[i]}', fontsize=12, fontweight='bold')

            # Set the title
            axes[i].set_title(f'{feature_names[i]}', fontsize=13, fontweight='bold', pad=10)

            # Add grid lines
            axes[i].grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)

            # Save the output artifact.
            text_str = f'Training:\n  MAE = {train_mae:.3f}\n  $R^{{2}}$ = {train_r2:.3f}\n\nTesting:\n  MAE = {test_mae:.3f}\n  $R^{{2}}$ = {test_r2:.3f}'
            axes[i].text(0.05, 0.95, text_str, transform=axes[i].transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                                   edgecolor='gray'))

            # Set axis limits
            buffer = 0.05 * (max_val - min_val)
            axes[i].set_xlim([min_val - buffer, max_val + buffer])
            axes[i].set_ylim([min_val - buffer, max_val + buffer])

            # Set tick locations
            axes[i].tick_params(axis='both', which='major', labelsize=10)

            # Add a legend
            axes[i].legend(loc='lower right', fontsize=9, framealpha=0.9)

        # Set the overall title
        plt.suptitle('Training and Test Set Prediction Comparison', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the output artifact.
        plot_path = os.path.join(self.results_dir, 'combined_predictions_vs_actual.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Prediction comparison plot saved to: {plot_path}")

        plt.show()

    def extract_decoder_for_future_use(self, output_path):
        """Extract the decoder for later use"""
        try:
            if self.final_trainer.model_builder is None:
                if not self.final_trainer.load_final_model():
                    return None

            # Get the decoder
            decoder = self.final_trainer.model_builder.get_decoder()

            # Save to the specified path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            decoder.save(output_path)
            print(f"Final-model decoder extracted and saved to: {output_path}")

            return decoder

        except Exception as e:
            print(f"Error extracting the decoder: {e}")
            return None

    def generate_final_report(self, train_results, test_results):
        """Generate the final evaluation report"""
        print("=" * 60)
        print("Final model evaluation report")
        print("=" * 60)

        print(f"Training-set size: {len(train_results['X_train_original'])} samples")
        print(f"Test-set size: {len(test_results['X_test_original'])} samples")

        print("=== Training-set metrics ===")
        print(f" MSE: {train_results['mse']:.6f}")
        print(f" RMSE: {train_results['rmse']:.6f}")
        print(f" Overall R²: {train_results['r2']:.6f}")
        for i, r2 in enumerate(train_results['r2_per_feature']):
            print(f" Feature {i + 1} R²: {r2:.6f}")

        print("=== Test-set metrics ===")
        print(f" MSE: {test_results['mse']:.6f}")
        print(f" RMSE: {test_results['rmse']:.6f}")
        print(f" Overall R²: {test_results['r2']:.6f}")
        for i, r2 in enumerate(test_results['r2_per_feature']):
            print(f" Feature {i + 1} R²: {r2:.6f}")

        # Save combined prediction data.
        combined_file = self.save_combined_predictions(train_results, test_results)

        # Generate comparison plots
        # Plot predictions for the training set.
        self.plot_predictions_vs_actual(
            train_results['X_train_original'],
            train_results['predictions'],
            "Training Prediction Results",
            "train_predictions_vs_actual.png",
            "Training"
        )

        # Plot predictions for the test set.
        self.plot_predictions_vs_actual(
            test_results['X_test_original'],
            test_results['predictions'],
            "Testing Prediction Results",
            "test_predictions_vs_actual.png",
            "Testing"
        )

        self.plot_combined_predictions(
            train_results['X_train_original'],
            train_results['predictions'],
            test_results['X_test_original'],
            test_results['predictions']
        )

        return {
            'train_results': train_results,
            'test_results': test_results,
            'combined_prediction_file': combined_file
        }


def save_hyperparameters(best_params, results_dir):
    """Save hyperparameters to a file"""
    # Ensure the output directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Build the hyperparameter-file path
    hyperparameter_file = os.path.join(results_dir, 'hyperparameter.txt')

    # Write hyperparameters to the file
    with open(hyperparameter_file, 'w', encoding='utf-8') as f:
        f.write("Hyperparameter Configuration\n")
        f.write("=" * 30 + "\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    print(f"Hyperparameters saved to: {hyperparameter_file}")

def main():
    """Main entry point"""
    print("=" * 60)
    print("Final model training and evaluation")
    print("=" * 60)

    # Load preprocessed data
    data_dict = load_preprocessed_data('processed_data_cv')
    if data_dict is None:
        print("Failed to load preprocessed data.")
        return

    X_train = data_dict['X_train']  # 100 training samples
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']  # 15 independent test samples
    y_test = data_dict['y_test']

    print(f"Training-set shape: {X_train.shape}")
    print(f"Test-set shape: {X_test.shape}")

    # Create the evaluator
    evaluator = CVEvaluation('final_model_results_03_500')

    # Set the selected hyperparameters based on the grid-search results
    best_params = {
        'learning_rate': 0.002,
        'dropout_rate': 0.15,
        'batch_size': 8,
        'activation': 'relu',
        'hidden_layers': [128, 64]
    }

    print(f"Selected hyperparameters: {best_params}")

    # Save hyperparameters to a file
    save_hyperparameters(best_params, evaluator.results_dir)

    # Train the final model using the selected hyperparameters.
    history, train_results = evaluator.final_trainer.train_final_model(
        X_train, y_train, best_params, epochs=500)

    # Evaluate the model on the independent test set.
    test_results = evaluator.evaluate_on_testset(X_test, y_test)

    if test_results is not None:
        # Generate the final report
        final_report = evaluator.generate_final_report(train_results, test_results)

        # Extract the decoder for downstream use
        decoder_path = os.path.join('extracted_models', 'final_decoder.h5')
        evaluator.extract_decoder_for_future_use(decoder_path)

        print("Final model training and evaluation completed.")
        print(f"Training predictions saved to: {train_results.get('prediction_file', 'N/A')}")
        print(f"Test predictions saved to: {test_results.get('prediction_file', 'N/A')}")
        print(f"Combined predictions saved to: {final_report.get('combined_prediction_file', 'N/A')}")


if __name__ == "__main__":
    main()