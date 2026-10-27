"""Prepare experimental data and five-fold cross-validation indices."""

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_and_prepare_data(raw_data_path, output_dir='processed_data_cv',
                          feature_columns=None, label_column=None,
                          test_size=15, n_splits=5, random_state=42):
    """Load the experimental dataset and prepare the fixed train/test split.

    Args:
        raw_data_path: Path to the original Excel dataset.
        output_dir: Directory for the saved arrays and cross-validation indices.
        feature_columns: Input feature names in the original Excel file.
        label_column: Shear-stress column name in the original Excel file.
        test_size: Number of independent test samples (15 in this study).
        n_splits: Number of cross-validation folds.
        random_state: Random seed used to shuffle the training folds.

    The test indices are selected deterministically using np.linspace.
    Five-fold cross-validation is applied only to the remaining training data.
    Standardization is performed later, separately within each training fold.
    """
    if feature_columns is None:
        feature_columns = ['Shear_rate(/s)', 'Temperature(℃)', 'Concentration(%)']
    if label_column is None:
        label_column = 'Shear_stress(Pas)'

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('Loading and preprocessing data for cross-validation')
    print('=' * 60)

    try:
        # Load the original experimental data.
        print('\n1. Loading the experimental dataset...')
        df = pd.read_excel(raw_data_path, engine='openpyxl')
        print(f'Data loaded. Shape: {df.shape}')

        # Check that the required columns are present.
        for col in feature_columns + [label_column]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing from the dataset")

        X = df[feature_columns].values
        y = df[label_column].values.reshape(-1, 1)

        print(f'Input features shape: {X.shape}')
        print(f'Shear-stress targets shape: {y.shape}')

        # Preserve the original deterministic 100/15 split.
        print('\n2. Splitting the dataset into training and test sets...')
        total_samples = len(X)
        test_indices = np.linspace(0, total_samples - 1, test_size, dtype=int)
        train_indices = np.setdiff1d(np.arange(total_samples), test_indices)

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        print('Data split completed.')
        print(f'   Training samples: {X_train.shape[0]}')
        print(f'   Test samples: {X_test.shape[0]}')

        # Prepare cross-validation indices from training samples only.
        print('\n3. Preparing cross-validation folds...')
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        cv_data = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            cv_data.append({
                'fold': fold + 1,
                'train_indices': train_idx,
                'val_indices': val_idx
            })

        # Save the fold indices for the training pipeline.
        print('\n4. Saving preprocessed data...')
        cv_info = []
        for fold_data in cv_data:
            cv_info.append({
                'fold': fold_data['fold'],
                'train_indices': str(fold_data['train_indices'].tolist()),
                'val_indices': str(fold_data['val_indices'].tolist())
            })
        cv_df = pd.DataFrame(cv_info)
        cv_df.to_excel(os.path.join(output_dir, 'cv_split_info.xlsx'), index=False)

        # Save the original, unstandardized subsets for reference.
        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()
        train_df.to_excel(os.path.join(output_dir, 'train_data_original.xlsx'), index=False)
        test_df.to_excel(os.path.join(output_dir, 'test_data_original.xlsx'), index=False)

        # Save arrays consumed by the cross-validation and evaluation scripts.
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'cv_data': cv_data,
            'feature_columns': feature_columns,
            'label_column': label_column,
            'train_indices': train_indices,
            'test_indices': test_indices
        }

        print('\n' + '=' * 60)
        print('Data preparation summary')
        print('=' * 60)
        print(f'Total samples: {df.shape[0]}')
        print(f'Training samples: {X_train.shape[0]}')
        print(f'Test samples: {X_test.shape[0]}')
        print(f'Cross-validation folds: {n_splits}')

        return result

    except Exception as e:
        print(f'Data preparation failed: {e}')
        return None


def load_preprocessed_data(output_dir='processed_data_cv'):
    """Load saved arrays and cross-validation fold indices."""
    try:
        X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(output_dir, 'y_test.npy'))

        cv_info = pd.read_excel(os.path.join(output_dir, 'cv_split_info.xlsx'))

        # Safely deserialize fold indices without evaluating Python code.
        cv_data = []
        for _, row in cv_info.iterrows():
            cv_data.append({
                'fold': row['fold'],
                'train_indices': np.array(ast.literal_eval(row['train_indices']), dtype=int),
                'val_indices': np.array(ast.literal_eval(row['val_indices']), dtype=int)
            })

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'cv_data': cv_data
        }

    except Exception as e:
        print(f'Failed to load preprocessed data: {e}')
        return None


if __name__ == '__main__':
    # The experimental Excel file is stored alongside this script.
    raw_data_path = Path(__file__).resolve().parent / 'experimental_data.xlsx'

    data_dict = load_and_prepare_data(
        raw_data_path,
        test_size=15,
        n_splits=5,
        output_dir='processed_data_cv'
    )

    if data_dict is not None:
        print('\nData loading and preprocessing completed.')
        print(f"Cross-validation folds: {len(data_dict['cv_data'])}")
