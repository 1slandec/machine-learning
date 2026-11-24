import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_HOME', str(Path(__file__).resolve().parent / 'data/models/.keras'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / 'data/models/.ultralytics'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from ultralytics import YOLO

data_dir = Path(__file__).resolve().parent / 'data'
yolo_data_dir = data_dir / 'yolo_dataset'
image_size = (224, 224)
batch_size = 16
epochs = 50
random_state = 42


def prepare_data():
    dataset_dir = data_dir / 'dataset'
    parts = {}
    for split in ('train', 'validation', 'test'):
        rows = []
        for folder, label in [('cats', 1), ('no_cats', 0)]:
            for path in sorted((dataset_dir / split / folder).glob('*.jpg')):
                rows.append({'filepath': str(path), 'label': label})
        part = pd.DataFrame(rows)
        parts[split] = part
        print(f'{split}: {len(part)} изображений, '
              f'с кошками — {int(part["label"].sum())}, '
              f'без кошек — {int((part["label"] == 0).sum())}')
    return parts


def prepare_yolo_data():
    config = yaml.safe_load((yolo_data_dir / 'data.yaml').read_text(encoding='utf-8'))
    return {
        split: pd.DataFrame({'filepath': [
            str(path) for path in sorted((yolo_data_dir / config[key]).glob('*.jpg'))
        ]})
        for split, key in [('train', 'train'), ('validation', 'val'), ('test', 'test')]
    }


def create_dataset(dataframe, shuffle=False, preprocess=False):
    def load_image(filepath, label):
        image = tf.io.decode_image(
            tf.io.read_file(filepath), channels=3, expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        if preprocess:
            image = tf.keras.applications.vgg16.preprocess_input(image)
        return image, tf.cast(label, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe['filepath'].to_numpy(), dataframe['label'].to_numpy(),
    )).map(load_image, num_parallel_calls=2).cache()
    if shuffle:
        dataset = dataset.shuffle(len(dataframe), seed=random_state)
    return dataset.batch(batch_size).prefetch(1)


def fit_model(model, train_ds, val_ds, labels=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy', metrics=['accuracy'],
    )
    return model.fit(
        train_ds, labels, validation_data=val_ds, batch_size=batch_size,
        epochs=epochs, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True,
        )],
    )


def evaluate_classifier(model, test_ds, test_df, name):
    y_test = test_df['label'].to_numpy()
    y_pred = (model.predict(test_ds, verbose=0).ravel() >= 0.5).astype(int)
    return {'Модель': name, 'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0)}


def train_classifiers(parts):
    train_ds = create_dataset(parts['train'], shuffle=True)
    val_ds = create_dataset(parts['validation'])
    test_ds = create_dataset(parts['test'])

    cnn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*image_size, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    fit_model(cnn, train_ds, val_ds)
    results = [evaluate_classifier(cnn, test_ds, parts['test'], 'CNN')]
    del cnn, train_ds, val_ds, test_ds
    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(random_state)
    base = tf.keras.applications.VGG16(
        weights='imagenet', include_top=False, input_shape=(*image_size, 3),
    )
    base.trainable = False
    extractor = tf.keras.Sequential([base, tf.keras.layers.GlobalAveragePooling2D()])
    train_features = extractor.predict(create_dataset(parts['train'], preprocess=True), verbose=0)
    val_features = extractor.predict(create_dataset(parts['validation'], preprocess=True), verbose=0)
    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(512,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    fit_model(
        head, train_features, (val_features, parts['validation']['label'].to_numpy()),
        labels=parts['train']['label'].to_numpy(),
    )
    vgg = tf.keras.Sequential([tf.keras.layers.Input(shape=(*image_size, 3)), extractor, head])
    results.append(evaluate_classifier(
        vgg, create_dataset(parts['test'], preprocess=True), parts['test'], 'VGG16',
    ))
    return pd.DataFrame(results)


def save_yolo_predictions(model, paths, name, size):
    examples = []
    for result in model.predict(
        source=[str(path) for path in paths][:6], stream=True,
        conf=0.25, imgsz=size, device='cpu', verbose=False,
    ):
        examples.append((Path(result.path).name, result.plot()[..., ::-1]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (filename, image) in zip(axes.ravel(), examples):
        ax.imshow(image)
        ax.set_title(filename.split('_jpg.rf.')[0], fontsize=9)
    fig.tight_layout()
    fig.savefig(data_dir / f'{name}.png', dpi=150)
    plt.close(fig)


def predict_cats(model, path):
    cat_id = next(index for index, name in model.names.items() if name == 'cat')
    return model.predict(
        source=str(path), classes=[cat_id], conf=0.25, imgsz=320,
        device='cpu', verbose=False,
    )[0]


def read_yolo_boxes(path, shape):
    label_path = path.parent.parent / 'labels' / path.with_suffix('.txt').name
    rows = np.array([
        list(map(float, line.split()))
        for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()
    ]).reshape(-1, 5)
    height, width = shape
    boxes = rows[:, 1:] * [width, height, width, height]
    return np.column_stack((boxes[:, :2] - boxes[:, 2:] / 2,
                            boxes[:, :2] + boxes[:, 2:] / 2))


def match_yolo_boxes(predictions, truth):
    intersection = np.maximum(
        np.minimum(predictions[:, None, 2:4], truth[None, :, 2:4])
        - np.maximum(predictions[:, None, :2], truth[None, :, :2]), 0,
    ).prod(axis=2)
    pred_area = np.maximum(predictions[:, 2:4] - predictions[:, :2], 0).prod(axis=1)
    true_area = np.maximum(truth[:, 2:4] - truth[:, :2], 0).prod(axis=1)
    ious = intersection / (pred_area[:, None] + true_area[None, :] - intersection + 1e-9)
    correct = np.zeros(len(predictions), dtype=bool)
    used = set()
    for index in np.argsort(-predictions[:, 4], kind='stable'):
        for target in np.argsort(-ious[index], kind='stable'):
            if ious[index, target] >= 0.5 and target not in used:
                correct[index] = True
                used.add(target)
                break
    best_iou = ious.max(axis=1) if len(truth) else np.zeros(len(predictions))
    return correct, best_iou


def collect_yolo_diagnostics(model, parts, name):
    image_rows, box_rows, threshold_rows = [], [], []
    for split, part in parts.items():
        print(f'YOLO diagnostics: {name}/{split}')
        counts = {value: dict(TP=0, FP=0, FN=0, background_FP=0)
                  for value in (0.25, 0.5, 0.9, 0.99999)}
        for filepath in part['filepath']:
            path = Path(filepath)
            result = predict_cats(model, path)
            predictions = result.boxes.data.cpu().numpy()
            truth = read_yolo_boxes(path, result.orig_shape)
            correct, best_iou = match_yolo_boxes(predictions, truth)
            tp = int(correct.sum())
            image_rows.append(dict(
                model=name, split=split, filepath=filepath, GT=len(truth),
                TP=tp, FP=len(predictions) - tp, FN=len(truth) - tp,
            ))
            for box, is_correct, overlap in zip(predictions, correct, best_iou):
                status = ('TP' if is_correct else 'duplicate' if overlap >= 0.5
                          else 'low_IoU' if overlap >= 0.1 else 'background')
                box_rows.append(dict(
                    model=name, split=split, filepath=filepath, confidence=float(box[4]),
                    x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]),
                    max_IoU=float(overlap), status=status,
                ))
            for threshold, count in counts.items():
                selected = predictions[:, 4] >= threshold
                selected_tp = int(correct[selected].sum())
                selected_fp = int(selected.sum()) - selected_tp
                count['TP'] += selected_tp
                count['FP'] += selected_fp
                count['FN'] += len(truth) - selected_tp
                count['background_FP'] += selected_fp if len(truth) == 0 else 0
        for threshold, count in counts.items():
            tp, fp, fn = count['TP'], count['FP'], count['FN']
            threshold_rows.append(dict(
                model=name, split=split, confidence=threshold, **count,
                Precision=tp / max(tp + fp, 1), Recall=tp / max(tp + fn, 1),
                F1=2 * tp / max(2 * tp + fp + fn, 1),
            ))
    return pd.DataFrame(image_rows), pd.DataFrame(box_rows, columns=[
        'model', 'split', 'filepath', 'confidence', 'x1', 'y1', 'x2', 'y2', 'max_IoU', 'status',
    ]), pd.DataFrame(threshold_rows)


def plot_yolo_examples(pretrained, trained, images, output_dir):
    errors = images[(images['model'] == 'after') & (images['split'] == 'test')]
    selected = pd.concat([
        errors[errors['GT'] > 0].sort_values(['FP', 'FN'], ascending=False).head(3),
        errors[errors['GT'] == 0].sort_values('FP', ascending=False).head(3),
    ])
    for number, row in enumerate(selected.itertuples(), 1):
        path = Path(row.filepath)
        before = predict_cats(pretrained, path)
        after = predict_cats(trained, path)
        truth = read_yolo_boxes(path, after.orig_shape)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax in axes:
            ax.imshow(after.orig_img[..., ::-1])
            ax.axis('off')
        for x1, y1, x2, y2 in truth:
            axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor='limegreen', linewidth=2))
        axes[0].set_title(f'Ground truth: {len(truth)} cats')
        for ax, result, title in zip(axes[1:], (before, after), ('Before', 'After')):
            predictions = result.boxes.data.cpu().numpy()
            correct, _ = match_yolo_boxes(predictions, truth)
            for box, is_correct in zip(predictions, correct):
                x1, y1, x2, y2, confidence, _ = box
                color = 'limegreen' if is_correct else 'red'
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=color, linewidth=1.5))
                ax.text(x1, max(y1, 12), f'{"TP" if is_correct else "FP"} {confidence:.5f}',
                        fontsize=7, color=color, bbox=dict(facecolor='white', alpha=0.8, pad=1))
            tp = int(correct.sum())
            ax.set_title(f'{title}: TP={tp}, FP={len(predictions) - tp}, FN={len(truth) - tp}')
        fig.suptitle(f'{path.stem.split("_jpg.rf.")[0]} | conf >= 0.25, IoU >= 0.5')
        fig.tight_layout()
        fig.savefig(output_dir / f'example_{number}.png', dpi=150)
        plt.close(fig)
    selected.to_csv(output_dir / 'examples.csv', index=False, encoding='utf-8-sig')


def save_yolo_diagnostics(pretrained, trained, parts, run_dir):
    output_dir = data_dir / 'yolo_diagnostics' / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    before = collect_yolo_diagnostics(pretrained, {'test': parts['test']}, 'before')
    after = collect_yolo_diagnostics(trained, parts, 'after')
    images, boxes, thresholds = [pd.concat(items, ignore_index=True)
                                for items in zip(before, after)]
    for name, table in [('images', images), ('boxes', boxes), ('thresholds', thresholds)]:
        table.to_csv(output_dir / f'{name}.csv', index=False, encoding='utf-8-sig')

    history = pd.read_csv(run_dir / 'results.csv')
    history.columns = history.columns.str.strip()
    best = history.loc[history['metrics/mAP50-95(B)'].idxmax()]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, loss in zip(axes[0], ('box_loss', 'cls_loss')):
        for split in ('train', 'val'):
            ax.plot(history['epoch'], history[f'{split}/{loss}'], label=split)
        ax.set_title(loss)
    for metric in ('mAP50', 'mAP50-95'):
        axes[1, 0].plot(history['epoch'], history[f'metrics/{metric}(B)'], label=metric)
    axes[1, 0].set_title('Validation mAP')
    axes[1, 1].plot(history['epoch'], history['lr/pg0'], label='learning rate')
    axes[1, 1].set_title('Learning rate')
    for ax in axes.ravel():
        ax.axvline(best['epoch'], color='gray', linestyle='--', label='best epoch')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'training.png', dpi=150)
    plt.close(fig)

    test_boxes = boxes[(boxes['model'] == 'after') & (boxes['split'] == 'test')]
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, mask, color in [('TP', test_boxes['status'] == 'TP', 'green'),
                               ('FP', test_boxes['status'] != 'TP', 'red')]:
        ax.hist(test_boxes.loc[mask, 'confidence'], bins=np.linspace(0.25, 1, 16),
                alpha=0.5, label=label, color=color)
    ax.set(xlabel='Confidence', ylabel='Number of boxes', title='Test: confidence of TP and FP')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'confidence.png', dpi=150)
    plt.close(fig)
    plot_yolo_examples(pretrained, trained, images, output_dir)

    test_thresholds = thresholds[thresholds['split'] == 'test']
    operating_point = thresholds[thresholds['confidence'] == 0.25]
    summary = ('YOLO: сопоставление рамок по убыванию confidence, IoU >= 0.5, '
               'каждая истинная рамка учитывается один раз.\n'
               'Исходная модель оценивается только по классу cat.\n'
               'TP — правильная рамка, FP — лишняя, FN — пропущенный объект.\n'
               'duplicate — повторная рамка с IoU >= 0.5; low_IoU — IoU от 0.1 до 0.5; '
               'background — IoU < 0.1.\n'
               'background_FP — рамки на изображениях без кошек.\n'
               'Пороги ниже применены к predict(imgsz=320) после стандартного NMS. '
               'Они не являются порогом, выбранным валидатором для mAP.\n'
               'Test используется для отчёта; настройки выбирайте по validation.\n\n')
    summary += (f'Эпох выполнено: {len(history)}; лучшая по validation mAP50-95: '
                f'{int(best["epoch"])} ({best["metrics/mAP50-95(B)"]:.4f}).\n\n')
    summary += 'Сравнение на test при разных порогах:\n'
    summary += test_thresholds.round(5).to_string(index=False) + '\n\n'
    summary += 'Train/validation/test при confidence >= 0.25:\n'
    summary += operating_point.round(4).to_string(index=False) + '\n\n'
    summary += 'Ошибки рамок after/test:\n'
    summary += test_boxes['status'].value_counts().to_string() + '\n'
    (output_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)
    print(f'Диагностика YOLO: {output_dir}')
    return operating_point[(operating_point['model'] == 'after')
                           & (operating_point['split'] == 'test')].iloc[0]


def train_yolo(parts):
    weights = data_dir / 'models/yolov8n.pt'
    pretrained = YOLO(str(weights))
    save_yolo_predictions(pretrained, parts['test']['filepath'], 'yolo_pretrained', 320)

    config = str(yolo_data_dir / 'data.yaml')
    training_model = YOLO(str(weights))
    result = training_model.train(
        data=config, epochs=epochs, patience=10, imgsz=320, batch=8,
        device='cpu', workers=0, seed=random_state, deterministic=True, amp=False,
        project=str(data_dir / 'models'), name='yolo', plots=False,
    )

    trained = YOLO(str(Path(result.save_dir) / 'weights/best.pt'))
    metrics = trained.val(data=config, split='test', imgsz=320, batch=8,
                          device='cpu', workers=0, plots=False,
                          project=str(result.save_dir), name='test')
    table = pd.DataFrame([{'mAP50': float(metrics.box.map50),
                           'mAP50-95': float(metrics.box.map)}])
    save_yolo_predictions(trained, parts['test']['filepath'], 'yolo_after_finetuning', 320)
    diagnostics = save_yolo_diagnostics(pretrained, trained, parts, Path(result.save_dir))
    for metric in ('Precision', 'Recall', 'FP', 'FN'):
        table[f'{metric}@0.25'] = diagnostics[metric]
    return table


def save_results(classification, detection, parts):
    text = 'Выборки: ' + ', '.join(
        f'{split}={len(part)}' for split, part in parts.items()
    ) + '.\n\n'
    text += 'CNN/VGG (Precision, Recall, F1 — для класса «Кошка»):\n'
    text += classification.round(4).to_string(index=False) + '\n\n'
    text += 'YOLOv8 (test; Precision/Recall/FP/FN при conf=0.25, IoU=0.5):\n'
    text += detection.round(4).to_string(index=False) + '\n'
    (data_dir / 'results.txt').write_text(text, encoding='utf-8')
    print(text)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(random_state)
    tf.config.experimental.enable_op_determinism()
    (data_dir / 'models').mkdir(parents=True, exist_ok=True)
    parts = prepare_data()
    classification = train_classifiers(parts)
    detection = train_yolo(prepare_yolo_data())
    save_results(classification, detection, parts)


if __name__ == '__main__':
    main()
