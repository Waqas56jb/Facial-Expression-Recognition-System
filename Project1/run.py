import os
import sys
import cv2
import random
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from training import ModifiedPerceptron
from data_processing import preprocess_image, IMG_SIZE


def log_final(msg: str):
    # in this last project i want logs to read like small sentences i can directly use in my report
    print(f"[final] {msg}")

# emotions used in this project
# i define all emotion labels here so i have one consistent list for whole project
CLASSES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
NUM_CLASSES = len(CLASSES)

BASE = "complete_dataset" if os.path.isdir("complete_dataset") else "."
# i set train validation and test folder paths based on base folder
train_dir = os.path.join(BASE, "Training")
val_dir = os.path.join(BASE, "Validation")
test_dir = os.path.join(BASE, "Testing")


def load_dataset(folder, use_clahe=True):
    """Load images and labels from folder."""
    # i read each image from folder and turn it into feature vector and label
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    X, y = [], []
    for label, emotion in enumerate(CLASSES):
        path = os.path.join(folder, emotion)
        if not os.path.isdir(path) and emotion == "Surprised":
            path = os.path.join(folder, "Surprise")
        if not os.path.isdir(path):
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            x = preprocess_image(img, use_clahe=use_clahe)
            X.append(x)
            y.append(label)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def one_hot_encode(labels, num_classes):
    """Convert label ids to one hot vectors."""
    # i create one hot matrix so each label becomes vector for loss function
    n = labels.shape[0]
    y = np.zeros((n, num_classes), dtype=np.float64)
    y[np.arange(n), labels] = 1.0
    return y


def cross_entropy_loss(probs, y_onehot, eps=1e-12, sample_weights=None):
    """Cross entropy loss, can use class weights."""
    # i compute average negative log probability so model learns correct class
    probs = np.clip(probs, eps, 1.0 - eps)
    per_sample = -np.sum(y_onehot * np.log(probs), axis=1)
    if sample_weights is not None:
        return np.average(per_sample, weights=sample_weights)
    return np.mean(per_sample)


def compute_class_weights(y_train, num_classes):
    """Give higher weight to rare classes."""
    # i give more weight to rare labels so training does not ignore them
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


def augment_batch_flip(X_batch, p_flip=0.5, img_size=48):
    """Random left right flip to make data a bit bigger."""
    # i flip some images so dataset grows and model generalizes better
    X = X_batch.copy()
    n = X.shape[0]
    flip = np.random.rand(n) < p_flip
    for i in range(n):
        if flip[i]:
            img = X[i].reshape(img_size, img_size)
            X[i] = np.fliplr(img).flatten()
    return X


def train_sgd(model, X_train, y_train, X_val, y_val, learning_rate, batch_size, epochs, verbose=True,
              momentum=0.9, weight_decay=1e-4, lr_decay=0.98, clip_grad_norm=1.0,
              class_weights=None, augment_flip=True):
    """Train the model with simple SGD style loop."""
    # i train model here with mini batches so i can handle many images and control learning rate and epochs
    n = X_train.shape[0]
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    lr = learning_rate

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_epoch = X_train[indices]
        y_epoch = y_train[indices]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_epoch[start:end].copy()
            y_batch = y_epoch[start:end]
            if augment_flip:
                X_batch = augment_batch_flip(X_batch, p_flip=0.5, img_size=IMG_SIZE)
            sample_weights = class_weights[y_batch] if class_weights is not None else None
            y_onehot = one_hot_encode(y_batch, model.output_size)
            # i get prediction from model for current batch and compute loss
            probs = model.forward(X_batch)
            loss = cross_entropy_loss(probs, y_onehot, sample_weights=sample_weights)
            epoch_loss += loss
            n_batches += 1
            model.sgd_step(X_batch, y_batch, lr, momentum=momentum, weight_decay=weight_decay,
                          clip_norm=clip_grad_norm, sample_weights=sample_weights)

        history["train_loss"].append(epoch_loss / n_batches)
        train_pred = model.predict(X_train)
        history["train_acc"].append(accuracy_score(y_train, train_pred))
        val_pred = model.predict(X_val)
        history["val_acc"].append(accuracy_score(y_val, val_pred))

        lr = lr * lr_decay

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train loss: {history['train_loss'][-1]:.4f} "
                  f"- Train acc: {history['train_acc'][-1]:.4f} - Val acc: {history['val_acc'][-1]:.4f}")
    return model, history


def build_model_from_params(params, input_size, output_size):
    # i build one model from parameter dictionary so genetic algorithm can test many settings
    return ModifiedPerceptron(
        input_size=input_size,
        hidden_sizes=params["hidden_sizes"],
        output_size=output_size,
        dropout=params.get("dropout", 0.4),
    )


def fitness(params, X_train, y_train, X_val, y_val, input_size, output_size, max_epochs=20, class_weights=None):
    """Fitness = validation accuracy. Returns (acc, model, history)."""
    # i train model with these parameters and return how good validation accuracy is
    model = build_model_from_params(params, input_size, output_size)
    model, history = train_sgd(
        model, X_train, y_train, X_val, y_val,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        epochs=min(params["epochs"], max_epochs),
        verbose=False,
        momentum=0.9, weight_decay=1e-4, lr_decay=0.98, clip_grad_norm=1.0,
        class_weights=class_weights, augment_flip=True,
    )
    pred = model.predict(X_val)
    return accuracy_score(y_val, pred), model, history


def genetic_optimize(X_train, y_train, X_val, y_val, input_size, output_size,
                    population_size=10, generations=4, max_epochs=20, class_weights=None):
    """Simple search over hyper parameters using a small GA loop."""
    # i use small genetic algorithm here so i can search learning rate hidden layers batch size and dropout
    lr_options = [0.01, 0.005, 0.001, 0.0005]
    hidden_options = [
        (512, 256, 128),
        (256, 128),
        (512, 128),
        (256, 256, 128),
        (128, 64),
        (256, 128, 64),
    ]
    batch_options = [16, 32, 64]
    epoch_options = [20, 30, 40]
    dropout_options = [0.3, 0.4, 0.5]

    def random_individual():
        return {
            "learning_rate": random.choice(lr_options),
            "hidden_sizes": random.choice(hidden_options),
            "batch_size": random.choice(batch_options),
            "epochs": random.choice(epoch_options),
            "dropout": random.choice(dropout_options),
        }

    def mutate(ind):
        i = random.randint(0, 4)
        if i == 0:
            ind["learning_rate"] = random.choice(lr_options)
        elif i == 1:
            ind["hidden_sizes"] = random.choice(hidden_options)
        elif i == 2:
            ind["batch_size"] = random.choice(batch_options)
        elif i == 3:
            ind["epochs"] = random.choice(epoch_options)
        else:
            ind["dropout"] = random.choice(dropout_options)
        return ind

    def crossover(a, b):
        return {
            "learning_rate": random.choice([a["learning_rate"], b["learning_rate"]]),
            "hidden_sizes": random.choice([a["hidden_sizes"], b["hidden_sizes"]]),
            "batch_size": random.choice([a["batch_size"], b["batch_size"]]),
            "epochs": random.choice([a["epochs"], b["epochs"]]),
            "dropout": random.choice([a["dropout"], b["dropout"]]),
        }

    population = [random_individual() for _ in range(population_size)]
    best_fitness = -1.0
    best_model = None
    best_params = None
    best_history = None

    total_evals = generations * population_size
    eval_count = 0
    for gen in range(generations):
        print(f"  Generation {gen+1}/{generations} (training {population_size} models, ~1-3 min each)...")
        print(f"  The {population_size} models (hyperparameter sets) in this generation:")
        for idx, params in enumerate(population):
            print(f"    Model {idx+1}: lr={params['learning_rate']}, hidden={params['hidden_sizes']}, "
                  f"batch={params['batch_size']}, epochs={params['epochs']}")
        scores_models = []
        for idx, params in enumerate(population):
            eval_count += 1
            print(f"    Training model {eval_count}/{total_evals} ...", end=" ", flush=True)
            acc, model, history = fitness(params, X_train, y_train, X_val, y_val,
                                           input_size, output_size, max_epochs, class_weights)
            print(f"val_acc={acc:.3f}", flush=True)
            scores_models.append((acc, model, params, history))
            if acc > best_fitness:
                best_fitness = acc
                best_model = model
                best_params = params
                best_history = history
        scores_models.sort(key=lambda x: x[0], reverse=True)
        elite = [p for (_, _, p, __) in scores_models[: population_size // 2]]
        new_pop = list(elite)
        while len(new_pop) < population_size:
            p1, p2 = random.sample(elite, 2)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop
        print(f"  Generation {gen+1}/{generations} done - Best validation accuracy: {best_fitness:.4f}")

    return best_model, best_params, best_fitness, best_history


def plot_training_curves(history, save_path=None):
    """summarize how this final network learns across epochs"""
    # in my last version i like to keep one neat plot that i can drop in report without editing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss (Cross-Entropy)")
    ax1.set_title("Training Loss vs Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is None:
        os.makedirs("insights", exist_ok=True)
        save_path = os.path.join("insights", "training_overview_final.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_confusion_matrix(cm, class_names, save_path=None):
    """visual check of how well final model separates every emotion class"""
    # for this final run i rely on this picture to talk about strengths and weak spots in each label
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix",
           ylabel="Actual", xlabel="Predicted")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path is None:
        os.makedirs("insights", exist_ok=True)
        save_path = os.path.join("insights", "confusion_insights_final.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    # i parse command line options so i can choose to run genetic algorithm or simple training
    parser = argparse.ArgumentParser(description="Train Facial Expression Recognition model")
    parser.add_argument("--no-ga", action="store_true", help="Skip GA; use default hyperparameters")
    parser.add_argument("--ga-generations", type=int, default=4)
    parser.add_argument("--ga-population", type=int, default=10)
    args = parser.parse_args()

    log_final("i am preparing dataset splits for my main final model")
    log_final(f"for this final run i use base directory {BASE}")
    X_train, y_train = load_dataset(train_dir)
    X_val, y_val = load_dataset(val_dir)
    X_test, y_test = load_dataset(test_dir)
    log_final(f"i have {X_train.shape[0]} train {y_val.shape[0]} val and {X_test.shape[0]} test images in final setup")
    log_final(f"in this final model each sample is mapped to {X_train.shape[1]} input features")

    input_size = X_train.shape[1]
    output_size = NUM_CLASSES
    class_weights = compute_class_weights(y_train, NUM_CLASSES)
    log_final(f"i computed balanced class weights for the final run {np.round(class_weights, 3)}")

    if not args.no_ga:
        log_final("i now run genetic algorithm to fine tune final hyperparameters")
        model, best_params, _, history = genetic_optimize(
            X_train, y_train, X_val, y_val, input_size, output_size,
            population_size=args.ga_population, generations=args.ga_generations,
            class_weights=class_weights,
        )
        log_final(f"for the final report i selected these hyperparameters {best_params}")
        log_final("i train the final chosen model for more epochs with learning rate decay")
        model, history = train_sgd(
            model, X_train, y_train, X_val, y_val,
            learning_rate=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            epochs=100,
            verbose=True,
            momentum=0.9, weight_decay=1e-4, lr_decay=0.97, clip_grad_norm=1.0,
            class_weights=class_weights, augment_flip=True,
        )
    else:
        log_final("ga is disabled so i directly train my default final architecture")
        # i build default network when i do not use genetic algorithm
        model = ModifiedPerceptron(
            input_size=input_size,
            hidden_sizes=(512, 256, 128),
            output_size=output_size,
            dropout=0.4,
        )
        model, history = train_sgd(model, X_train, y_train, X_val, y_val,
                                  learning_rate=0.002, batch_size=32, epochs=60,
                                  momentum=0.9, weight_decay=1e-4, lr_decay=0.97, clip_grad_norm=1.0,
                                  class_weights=class_weights, augment_flip=True)

    if history is not None and len(history.get("train_loss", [])) > 0:
        log_final("i save training_overview_final plot to use it directly in my report")
        plot_training_curves(history)

    log_final("i now look at validation metrics for this final network")
    val_pred = model.predict(X_val)
    log_final(f"final model validation accuracy is {accuracy_score(y_val, val_pred):.4f}")

    log_final("i am evaluating the final model on the test set for the summary section")
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="macro", zero_division=0)
    rec = recall_score(y_test, pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, pred)
    log_final(f"final test accuracy is {acc:.4f}")
    log_final(f"final macro precision is {prec:.4f}")
    log_final(f"final macro recall is {rec:.4f}")
    log_final(f"final macro f1 score is {f1:.4f}")
    log_final(f"i include this confusion matrix for final model discussion\n{cm}")

    log_final("i generate confusion_insights_final image for my appendix")
    plot_confusion_matrix(cm, CLASSES)

    log_final("i am saving the final weights that i will use for demo")
    save_dict = {
        "model_params": model.get_params_dict(),
        "CLASSES": CLASSES,
        "IMG_SIZE": IMG_SIZE,
    }
    with open("weights.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    log_final("weights.pkl is stored for the final model")

    log_final("i test a couple of random images to show example predictions from the final system")
    test_images = []
    for emotion in CLASSES:
        folder = os.path.join(test_dir, emotion)
        if os.path.isdir(folder):
            for img in os.listdir(folder):
                test_images.append(os.path.join(folder, img))
    if test_images:
        random_imgs = random.sample(test_images, min(2, len(test_images)))
        for i, img_path in enumerate(random_imgs):
            true_label = os.path.basename(os.path.dirname(img_path))
            pred, conf = predict_image(img_path, model=model, load_from_disk=False)
            if pred is not None:
                log_final(f"example {i+1} true={true_label} predicted={pred} confidence={conf:.4f}")

    log_final("for real time mode with the final model i run: python real_time_emotion.py")


def predict_image(path, model=None, load_from_disk=True):
    """Final application: load image -> preprocess -> model prediction -> emotion label."""
    # i use this helper when i want to test only one image and see predicted emotion and confidence
    if model is None and load_from_disk:
        # i load already trained best weights from weights.pkl saved by this script
        with open("weights.pkl", "rb") as f:
            save_dict = pickle.load(f)
        model = ModifiedPerceptron(
            input_size=save_dict["model_params"]["input_size"],
            hidden_sizes=tuple(save_dict["model_params"]["hidden_sizes"]),
            output_size=save_dict["model_params"]["output_size"],
            dropout=save_dict["model_params"].get("dropout", 0.5),
        )
        model.set_params_dict(save_dict["model_params"])
        CLASSES_LOCAL = save_dict["CLASSES"]
    else:
        CLASSES_LOCAL = CLASSES
    img = cv2.imread(path)
    if img is None:
        return None, 0.0
    x = preprocess_image(img).reshape(1, -1)
    probs = model.predict_proba(x)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return CLASSES_LOCAL[pred], conf


if __name__ == "__main__":
    # i allow quick single image prediction mode if user passes only image path
    if len(sys.argv) == 2 and not sys.argv[1].startswith("--"):
        img_path = sys.argv[1]
        pred, conf = predict_image(img_path, model=None, load_from_disk=True)
        if pred is None:
            print("Could not load image:", img_path)
            sys.exit(1)
        print(f"Image: {img_path}")
        print(f"Predicted emotion: {pred}")
        print(f"Confidence: {conf:.4f}")
    else:
        main()
