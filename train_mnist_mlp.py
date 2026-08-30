import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mnist-mlp-experiments")

# oad MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X = X / 255.0  # Normalize pixel values

X_train, X_val, y_train, y_val = train_test_split(
    X[:10000], y[:10000], test_size=0.2, random_state=42, stratify=y[:10000]
)

#  6 Configurations (Varying learning rate & hidden layer architecture)
configs = [
    {"lr": 0.001, "hidden_layers": (64,),       "run_name": "run_1_lr001_hidden64"},
    {"lr": 0.01,  "hidden_layers": (64,),       "run_name": "run_2_lr01_hidden64"},
    {"lr": 0.1,   "hidden_layers": (64,),       "run_name": "run_3_lr1_hidden64"},
    {"lr": 0.001, "hidden_layers": (128, 64),   "run_name": "run_4_lr001_hidden128_64"},
    {"lr": 0.01,  "hidden_layers": (128, 64),   "run_name": "run_5_lr01_hidden128_64"},
    {"lr": 0.1,   "hidden_layers": (128, 64),   "run_name": "run_6_lr1_hidden128_64"},
]

# Execute 6 runs
for i, cfg in enumerate(configs, 1):
    with mlflow.start_run(run_name=cfg["run_name"]):
        # Exact logging statements
        mlflow.log_param("learning_rate", cfg["lr"])
        mlflow.log_param("hidden_layer_sizes", str(cfg["hidden_layers"]))
        mlflow.log_param("max_iter", 20)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("random_state", 42)
        
        # Train MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layers"],
            learning_rate_init=cfg["lr"],
            max_iter=20,
            batch_size=64,
            random_state=42,
            early_stopping=False
        )
        mlp.fit(X_train, y_train)
        
        # Metrics
        train_loss = float(mlp.loss_)
        val_preds = mlp.predict(X_val)
        val_accuracy = float(accuracy_score(y_val, val_preds))
        
        # Log Metrics & Model
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.sklearn.log_model(
    sk_model=mlp,
    artifact_path="model",
    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
)
        
        print(f"[{i}/6] {cfg['run_name']} -> Train Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f}")

print("\nAll 6 runs completed and logged to MLflow successfully!")