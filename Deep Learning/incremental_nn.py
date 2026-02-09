"""
Group Assignment 1: Incremental Learning on Large Data
Feed-forward Neural Network with Incremental Learning

This script implements:
- 3 hidden layer neural network with sigmoid activation
- Incremental (mini-batch) learning for large datasets
- Learning curve visualization
- Variable importance analysis
- Partial dependence plots
- RAM and time monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import os
from collections import deque

# Set random seed for reproducibility
np.random.seed(42)


class IncrementalNeuralNetwork:
    """
    Feed-forward neural network with 3 hidden layers and sigmoid activation.
    Supports incremental learning (online/mini-batch training).
    """

    def __init__(self, input_size, hidden_sizes=(64, 32, 16), output_size=1, learning_rate=0.01):
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            hidden_sizes: Tuple of hidden layer sizes (default: 64, 32, 16)
            output_size: Number of outputs (1 for regression)
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = []

        # Build layer sizes: input -> hidden1 -> hidden2 -> hidden3 -> output
        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]

        # Initialize weights and biases using Xavier initialization
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for better convergence with sigmoid
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append({'W': W, 'b': b})

        # Track training metrics
        self.mse_history = []
        self.instances_trained = 0

    def sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input data (batch_size, input_size)

        Returns:
            Output prediction
        """
        self.activations = [X]

        current = X
        for i, layer in enumerate(self.layers):
            z = np.dot(current, layer['W']) + layer['b']

            # Apply sigmoid to hidden layers, linear for output
            if i < len(self.layers) - 1:
                current = self.sigmoid(z)
            else:
                current = z  # Linear output for regression

            self.activations.append(current)

        return current

    def backward(self, X, y, output):
        """
        Backward pass (backpropagation).

        Args:
            X: Input data
            y: True labels
            output: Network output from forward pass
        """
        m = X.shape[0]

        # Output layer error (MSE derivative)
        delta = output - y.reshape(-1, 1)

        # Backpropagate through layers
        for i in range(len(self.layers) - 1, -1, -1):
            # Gradient for weights and biases
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Update weights and biases
            self.layers[i]['W'] -= self.learning_rate * dW
            self.layers[i]['b'] -= self.learning_rate * db

            # Propagate error to previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.layers[i]['W'].T) * self.sigmoid_derivative(self.activations[i])

    def train_batch(self, X, y):
        """
        Train on a single mini-batch.

        Args:
            X: Input batch
            y: Target batch

        Returns:
            MSE for this batch
        """
        output = self.forward(X)
        mse = np.mean((output.flatten() - y) ** 2)
        self.backward(X, y, output)

        self.instances_trained += X.shape[0]
        self.mse_history.append(mse)

        return mse

    def predict(self, X):
        """Make predictions without updating weights."""
        current = X
        for i, layer in enumerate(self.layers):
            z = np.dot(current, layer['W']) + layer['b']
            if i < len(self.layers) - 1:
                current = self.sigmoid(z)
            else:
                current = z
        return current.flatten()


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def compute_r2(y_true, y_pred):
    """Compute R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_moving_average(data, window=1000):
    """Compute moving average of a list."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def calculate_variable_importance(model, X_sample, y_sample, feature_names):
    """
    Calculate variable importance using permutation importance.

    Args:
        model: Trained neural network
        X_sample: Sample of input data
        y_sample: Corresponding targets
        feature_names: List of feature names

    Returns:
        Dictionary of feature importances
    """
    # Baseline prediction
    baseline_pred = model.predict(X_sample)
    baseline_mse = np.mean((baseline_pred - y_sample) ** 2)

    importances = {}

    for i, name in enumerate(feature_names):
        # Permute feature i
        X_permuted = X_sample.copy()
        np.random.shuffle(X_permuted[:, i])

        # Predict with permuted feature
        permuted_pred = model.predict(X_permuted)
        permuted_mse = np.mean((permuted_pred - y_sample) ** 2)

        # Importance = increase in MSE
        importances[name] = permuted_mse - baseline_mse

    return importances


def partial_dependence(model, X_sample, feature_idx, feature_name, grid_resolution=50):
    """
    Calculate partial dependence for a feature.

    Args:
        model: Trained neural network
        X_sample: Sample of input data
        feature_idx: Index of feature to analyze
        feature_name: Name of the feature
        grid_resolution: Number of points in the grid

    Returns:
        grid_values, partial_dependence_values
    """
    # Create grid of values for this feature
    feature_values = X_sample[:, feature_idx]
    grid = np.linspace(np.min(feature_values), np.max(feature_values), grid_resolution)

    pd_values = []

    for val in grid:
        X_temp = X_sample.copy()
        X_temp[:, feature_idx] = val
        predictions = model.predict(X_temp)
        pd_values.append(np.mean(predictions))

    return grid, np.array(pd_values)


def train_incremental(train_file, model, batch_size=32, epochs=1,
                      feature_cols=['sku', 'price', 'order', 'duration', 'category'],
                      target_col='quantity'):
    """
    Train the model incrementally by reading data in chunks.

    Args:
        train_file: Path to training CSV file
        model: Neural network model
        batch_size: Mini-batch size
        epochs: Number of passes through the data
        feature_cols: List of feature column names
        target_col: Target column name
    """
    print(f"Starting incremental training...")
    print(f"Batch size: {batch_size}")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")

    start_time = time.time()
    memory_readings = [get_memory_usage()]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Read data in chunks
        chunk_iterator = pd.read_csv(train_file, chunksize=batch_size)

        for chunk_num, chunk in enumerate(chunk_iterator):
            # Extract features and target
            X = chunk[feature_cols].values.astype(np.float32)
            y = chunk[target_col].values.astype(np.float32)

            # Train on this batch
            mse = model.train_batch(X, y)

            # Track memory periodically
            if chunk_num % 1000 == 0:
                memory_readings.append(get_memory_usage())
                print(f"  Chunk {chunk_num}, Instances: {model.instances_trained}, "
                      f"MSE: {mse:.4f}, Memory: {memory_readings[-1]:.2f} MB")

    training_time = time.time() - start_time

    return {
        'training_time': training_time,
        'memory_readings': memory_readings,
        'max_memory': max(memory_readings),
        'avg_memory': np.mean(memory_readings)
    }


def evaluate_model(model, test_file, feature_cols, target_col, has_header=False):
    """
    Evaluate model on test data.

    Args:
        model: Trained model
        test_file: Path to test CSV file
        feature_cols: Feature column names/indices
        target_col: Target column name/index
        has_header: Whether the test file has a header
    """
    print("\nEvaluating on test data...")

    # Read test data
    if has_header:
        test_df = pd.read_csv(test_file)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)
    else:
        # Test file has no header - use column indices
        # Columns: sku, price, quantity, order, duration, category
        test_df = pd.read_csv(test_file, header=None)
        test_df.columns = ['sku', 'price', 'quantity', 'order', 'duration', 'category']
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = compute_r2(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test
    }


def plot_learning_curve(mse_history, output_path='learning_curve.png', window=1000):
    """Plot the learning curve (instances vs moving average MSE)."""
    plt.figure(figsize=(12, 6))

    # Compute moving average
    ma_mse = compute_moving_average(mse_history, window)
    instances = np.arange(window, len(mse_history) + 1) if len(mse_history) >= window else np.arange(1, len(mse_history) + 1)

    if len(mse_history) >= window:
        plt.plot(instances, ma_mse, 'b-', linewidth=0.5)
    else:
        plt.plot(instances, mse_history, 'b-', linewidth=0.5)

    plt.xlabel('Number of Instances Learned', fontsize=12)
    plt.ylabel(f'Moving Average MSE (window={window})', fontsize=12)
    plt.title('Learning Curve: Incremental Neural Network Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Learning curve saved to {output_path}")


def plot_variable_importance(importances, output_path='variable_importance.png'):
    """Plot variable importances."""
    plt.figure(figsize=(10, 6))

    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_features)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    bars = plt.barh(range(len(features)), values, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance (Increase in MSE when permuted)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Variable Importance (Permutation Importance)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Variable importance plot saved to {output_path}")


def plot_partial_dependence_all(model, X_sample, feature_names, output_path='partial_dependence.png'):
    """Plot partial dependence for all features."""
    n_features = len(feature_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        grid, pd_vals = partial_dependence(model, X_sample, i, name)
        ax.plot(grid, pd_vals, 'b-', linewidth=2)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Partial Dependence', fontsize=11)
        ax.set_title(f'Partial Dependence: {name}', fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide extra subplot if odd number of features
    if n_features < len(axes):
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)

    plt.suptitle('Partial Dependence Plots', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Partial dependence plots saved to {output_path}")


def main():
    """Main function to run the entire pipeline."""
    print("=" * 60)
    print("Group Assignment 1: Incremental Learning on Large Data")
    print("=" * 60)

    # Configuration
    TRAIN_FILE = 'pricing.csv'
    TEST_FILE = 'pricing_test.csv'
    FEATURE_COLS = ['sku', 'price', 'order', 'duration', 'category']
    TARGET_COL = 'quantity'
    BATCH_SIZE = 64
    EPOCHS = 3
    HIDDEN_SIZES = (64, 32, 16)
    LEARNING_RATE = 0.001

    # Record initial memory
    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")

    # Initialize model
    print(f"\nInitializing neural network...")
    print(f"Architecture: {len(FEATURE_COLS)} -> {HIDDEN_SIZES} -> 1")
    print(f"Activation: Sigmoid (hidden layers), Linear (output)")

    model = IncrementalNeuralNetwork(
        input_size=len(FEATURE_COLS),
        hidden_sizes=HIDDEN_SIZES,
        output_size=1,
        learning_rate=LEARNING_RATE
    )

    # Train incrementally
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)

    train_stats = train_incremental(
        TRAIN_FILE, model,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL
    )

    print(f"\nTraining completed!")
    print(f"Total training time: {train_stats['training_time']:.2f} seconds")
    print(f"Max memory usage: {train_stats['max_memory']:.2f} MB")
    print(f"Average memory usage: {train_stats['avg_memory']:.2f} MB")

    # Evaluate on test data
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)

    eval_results = evaluate_model(
        model, TEST_FILE,
        FEATURE_COLS, TARGET_COL,
        has_header=False
    )

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Learning curve
    plot_learning_curve(model.mse_history, 'learning_curve.png')

    # Variable importance (using test data sample)
    sample_size = min(5000, len(eval_results['X_test']))
    sample_idx = np.random.choice(len(eval_results['X_test']), sample_size, replace=False)
    X_sample = eval_results['X_test'][sample_idx]
    y_sample = eval_results['y_test'][sample_idx]

    print("\nCalculating variable importance...")
    importances = calculate_variable_importance(model, X_sample, y_sample, FEATURE_COLS)
    plot_variable_importance(importances, 'variable_importance.png')

    # Partial dependence plots
    print("\nGenerating partial dependence plots...")
    plot_partial_dependence_all(model, X_sample, FEATURE_COLS, 'partial_dependence.png')

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model Architecture: {len(FEATURE_COLS)} -> {HIDDEN_SIZES} -> 1")
    print(f"Activation: Sigmoid")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Total Instances Trained: {model.instances_trained}")
    print(f"Training Time: {train_stats['training_time']:.2f} seconds")
    print(f"Max Memory Usage: {train_stats['max_memory']:.2f} MB")
    print(f"Test R²: {eval_results['r2']:.4f}")
    print(f"Test RMSE: {eval_results['rmse']:.4f}")

    print("\nVariable Importances:")
    for name, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")

    print("\nOutput files generated:")
    print("  - learning_curve.png")
    print("  - variable_importance.png")
    print("  - partial_dependence.png")

    return model, eval_results, train_stats


if __name__ == "__main__":
    model, eval_results, train_stats = main()
