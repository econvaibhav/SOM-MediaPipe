"""Train the MLP architectures with separate train/validation/test splits.

Existing course models are not overwritten. Rows lack participant or session IDs,
so the reported metrics are an exploratory row-level evaluation.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def train(task='keypoint', dataset=None, output=None, epochs=200, seed=42):
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    if task not in ('keypoint', 'point_history'):
        raise ValueError('Choose keypoint or point_history')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    folder = ROOT / 'model' / f'{task}_classifier'
    dataset = Path(dataset) if dataset else folder / f'{task}.csv'
    output = Path(output) if output else ROOT / 'artifacts' / task
    if output.resolve() == folder.resolve() or folder.resolve() in output.resolve().parents:
        raise ValueError('Export to artifacts first; do not overwrite the bundled course models')
    with (folder / f'{task}_classifier_label.csv').open(encoding='utf-8-sig') as f:
        labels = [r[0] for r in csv.reader(f) if r]
    feature_count, hidden = (42, 20) if task == 'keypoint' else (32, 24)
    data = np.loadtxt(dataset, delimiter=',', dtype=np.float32, ndmin=2)
    if data.shape[1] != feature_count + 1 or not np.isfinite(data).all():
        raise ValueError(f'CSV requires label + {feature_count} finite features')
    # Avoid literal duplicate rows crossing the split, while retaining a count for reporting.
    original_rows = len(data)
    _, unique_indices = np.unique(data, axis=0, return_index=True)
    data = data[np.sort(unique_indices)]
    X, raw_y = data[:, 1:], data[:, 0]
    y = raw_y.astype(np.int64)
    if not np.array_equal(raw_y, y) or set(y) != set(range(len(labels))):
        raise ValueError('Dataset classes must match every label in the label CSV')
    if np.bincount(y).min() < 10:
        raise ValueError('Each class needs at least 10 distinct samples for the stratified split')
    ids = np.arange(len(y))
    development, test = train_test_split(ids, test_size=0.2, stratify=y, random_state=seed)
    training, validation = train_test_split(development, test_size=0.25,
                                           stratify=y[development], random_state=seed)
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(feature_count,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dropout(0.4 if task == 'keypoint' else 0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(len(labels), activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    output.mkdir(parents=True, exist_ok=True)
    model.fit(X[training], y[training], validation_data=(X[validation], y[validation]),
              epochs=epochs, batch_size=128, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                                         restore_best_weights=True)])
    predictions = np.argmax(model.predict(X[test], verbose=0), axis=1)
    report = classification_report(y[test], predictions, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True, zero_division=0)
    report['evaluation_note'] = 'Exploratory row split, not held-out people or sessions; near-duplicate frames may remain.'
    report['rows'] = {'original': original_rows, 'duplicates_removed': original_rows - len(data),
                      'train': len(training), 'validation': len(validation), 'test': len(test)}
    report['seed'] = seed
    report['confusion_matrix'] = confusion_matrix(y[test], predictions, labels=list(range(len(labels)))).tolist()
    model.save(output / f'{task}_classifier.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    (output / f'{task}_classifier.tflite').write_bytes(converter.convert())
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / 'split_indices.npz', original_row_indices=np.sort(unique_indices),
                        train=training, validation=validation, test=test)
    print(f'Exported to {output.resolve()}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['keypoint', 'point_history'], default='keypoint')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
