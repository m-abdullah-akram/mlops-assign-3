"""Task 2.7 - Keras Tuner random search over lr, dropout and number of unfrozen layers."""
import argparse
import json

import keras_tuner as kt
import tensorflow as tf

from visionguard import config, data, models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="mobilenetv2", choices=list(models.BACKBONES))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(config.SEED)

    def build(hp):
        lr = hp.Float("lr", 1e-5, 1e-2, sampling="log")
        dropout = hp.Float("dropout", 0.1, 0.6, step=0.1)
        n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 30, 50])
        model = models.build_transfer(args.backbone, dropout, args.img_size)
        models.unfreeze(model, args.backbone, n_unfrozen)
        return models.compile_model(model, lr)

    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=args.trials, seed=config.SEED,
                            directory=str(config.OUTPUT_DIR / "tuner"), project_name="visionguard",
                            overwrite=True)
    train_ds = data.make_dataset("train", args.img_size, 32, training=True)
    val_ds = data.make_dataset("val", args.img_size, 32)
    tuner.search(train_ds, validation_data=val_ds, epochs=args.epochs,
                 callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=2)])
    rows = []
    for t in tuner.oracle.get_best_trials(args.trials):
        rows.append({"trial": t.trial_id, "val_loss": t.score, **t.hyperparameters.values})
    out = config.REPORTS_DIR / "tuner_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print("Best hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    main()
