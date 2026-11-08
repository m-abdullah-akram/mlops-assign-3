"""Train a baseline CNN or a transfer-learning model, evaluate once on test, log to MLflow.

Examples
  python -m visionguard.train --model baseline --run-name baseline --tracking manual
  python -m visionguard.train --model transfer --backbone mobilenetv2 --lr 1e-3 --lr-finetune 1e-5
  python -m visionguard.train --model transfer --tracking autolog --run-name tf_autolog
  python -m visionguard.train --model transfer --register   # creates a new registry version
"""
import argparse
import json
import shutil
import subprocess
import sys
import time

import mlflow
import numpy as np
import tensorflow as tf

from visionguard import config, data, evaluate, models, tracking


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    ap.add_argument("--backbone", choices=list(models.BACKBONES), default="mobilenetv2")
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (stage 1 / baseline)")
    ap.add_argument("--lr-finetune", type=float, default=1e-5, help="stage 2 learning rate")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--unfreeze-layers", type=int, default=None,
                    help="layers to unfreeze in stage 2 (default: the last block)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--beta", type=float, default=1.0, help="F-beta used to pick the threshold")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--tracking", choices=["none", "manual", "autolog"], default="manual")
    ap.add_argument("--register", action="store_true", help="register the model in MLflow registry")
    ap.add_argument("--registered-name", default=f"{config.PROJECT_NAME}_model")
    return ap.parse_args(argv)


def make_callbacks(ckpt_path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def fit_stage(model, train_ds, val_ds, epochs, out_dir, stage, args, offset):
    callbacks = make_callbacks(out_dir / f"best_stage{stage}.keras")
    if args.tracking == "manual":
        callbacks.append(tracking.MlflowEpochCallback(stage, offset))
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    (out_dir / f"trainable_layers_stage{stage}.txt").write_text(
        "\n".join(models.trainable_layer_names(model)))
    print(f"[stage {stage}] best epoch (lowest val_loss): {best_epoch}")
    return hist.history, best_epoch


def predict(model, ds):
    return model.predict(ds, verbose=0).ravel()


def main(argv=None):
    args = parse_args(argv)
    tf.keras.utils.set_random_seed(args.seed)
    print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

    run_name = args.run_name or f"{args.model}-{args.backbone}-lr{args.lr}-{int(time.time())}"
    out_dir = config.OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = data.make_dataset("train", args.img_size, args.batch_size, True, not args.no_augment,
                                 args.seed)
    val_ds = data.make_dataset("val", args.img_size, args.batch_size)
    test_ds = data.make_dataset("test", args.img_size, args.batch_size)

    use_mlflow = args.tracking != "none"
    if use_mlflow:
        print("MLflow tracking URI:", tracking.setup_mlflow())
        run_ctx = mlflow.start_run(run_name=run_name)
    else:
        from contextlib import nullcontext
        run_ctx = nullcontext()

    with run_ctx:
        if use_mlflow:
            mlflow.set_tags({**tracking.standard_tags(args.seed), "logging_mode": args.tracking})
            mlflow.log_params({k: v for k, v in vars(args).items() if k != "register"})
            if args.tracking == "autolog":
                mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)

        histories, best_epochs = [], []
        if args.model == "baseline":
            model = models.build_baseline(args.img_size, args.dropout)
            models.compile_model(model, args.lr)
            (out_dir / "model_summary.txt").write_text(_summary(model))
            h, b = fit_stage(model, train_ds, val_ds, args.epochs, out_dir, 1, args, 0)
            histories.append(h); best_epochs.append(b)
        else:
            model = models.build_transfer(args.backbone, args.dropout, args.img_size)
            models.compile_model(model, args.lr)
            (out_dir / "model_summary.txt").write_text(_summary(model))
            h, b = fit_stage(model, train_ds, val_ds, args.epochs, out_dir, 1, args, 0)
            histories.append(h); best_epochs.append(b)
            if args.unfreeze_layers != 0:     # stage 2: fine-tune with a much smaller LR
                models.unfreeze(model, args.backbone, args.unfreeze_layers)
                models.compile_model(model, args.lr_finetune)   # must recompile after changing trainable
                h, b = fit_stage(model, train_ds, val_ds, args.epochs_finetune, out_dir, 2, args,
                                 len(histories[0]["loss"]))
                histories.append(h); best_epochs.append(b)

        # ---- threshold on validation, ONE evaluation on test ----
        y_val = data.load_split("val")[1]
        y_test = data.load_split("test")[1]
        test_paths = data.load_split("test")[0]
        p_val, p_test = predict(model, val_ds), predict(model, test_ds)
        threshold = evaluate.choose_threshold(y_val, p_val, args.beta)
        metrics = evaluate.compute_metrics(y_test, p_test, threshold)
        metrics["best_epochs"] = best_epochs
        print(json.dumps(metrics, indent=2))

        evaluate.plot_curves(histories, best_epochs, out_dir / "curves.png", run_name)
        evaluate.plot_confusion(y_test, p_test, threshold, out_dir / "confusion_matrix.png")
        evaluate.plot_roc(y_test, p_test, out_dir / "roc_curve.png", args.model)
        wrong = evaluate.misclassified_grid(test_paths, y_test, p_test, threshold,
                                            out_dir / "misclassified.png")
        metrics["misclassified_test_indices"] = wrong
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        model.save(out_dir / "model.keras")
        (out_dir / "requirements_frozen.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        shutil.copytree(config.SPLIT_DIR, out_dir / "splits", dirs_exist_ok=True)

        if use_mlflow:
            mlflow.log_param("threshold", round(threshold, 4))
            mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            for i, b in enumerate(best_epochs, 1):
                mlflow.log_metric(f"best_epoch_stage{i}", b)
            mlflow.log_artifacts(str(out_dir), artifact_path="run_files")   # plots, splits, pip list
            mlflow.log_artifact(str(out_dir / "model.keras"), artifact_path="model_file")
            mlflow.tensorflow.log_model(
                model, artifact_path="model",
                registered_model_name=args.registered_name if args.register else None)
    return metrics


def _summary(model):
    lines = []
    model.summary(print_fn=lines.append, expand_nested=False)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
