import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_HOME', str(Path(__file__).resolve().parent / 'data/models/.keras'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / 'data/models/.ultralytics'))

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent.parent
data_dir = Path(__file__).resolve().parent / 'data'
class_names = ['Остальное', 'Кошка']
image_size = (224, 224)
batch_size = 16
epochs = 50
random_state = 42


def prepare_data():
    rows = []
    for folder, label in [('cats', 1), ('no_cats', 0)]:
        for path in sorted((project_dir / folder).glob('*.jpg')):
            rows.append({'filename': path.name, 'filepath': str(path),
                         'folder': folder, 'label': label})
    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=random_state,
    )
    parts = {'train': train_df, 'validation': val_df, 'test': test_df}
    for split, part in parts.items():
        for row in part.itertuples():
            folder = data_dir / 'dataset' / split / row.folder
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.filepath, folder / row.filename)
            if row.label == 1 and not (folder / Path(row.filename).with_suffix('.txt')).is_file():
                raise FileNotFoundError(f'Нет разметки для {row.filename}')
        print(f'{split}: {len(part)} изображений')
    config = {'train': 'train', 'val': 'validation', 'test': 'test', 'names': ['cat']}
    (data_dir / 'dataset/data.yaml').write_text(
        yaml.safe_dump(config), encoding='utf-8',
    )
    return parts


def create_dataset(dataframe, shuffle=False, preprocess=False):
    def load_image(filepath, label):
        image = tf.io.decode_jpeg(tf.io.read_file(filepath), channels=3)
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
    print(f'\n{name}:')
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    return {'Модель': name, 'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0)}


def train_classifiers(parts):
    train_ds = create_dataset(parts['train'], shuffle=True)
    val_ds = create_dataset(parts['validation'])
    test_ds = create_dataset(parts['test'])

    # 1.2. Простая CNN, как в семинарском ноутбуке.
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
    cnn.save(data_dir / 'models/cnn.keras')
    results = [evaluate_classifier(cnn, test_ds, parts['test'], 'CNN')]
    del cnn, train_ds, val_ds, test_ds
    tf.keras.backend.clear_session()

    # 1.3. VGG16 с замороженной базой ImageNet и новой головой.
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
    # На вход сохранённой VGG16 подаётся результат vgg16.preprocess_input.
    vgg.save(data_dir / 'models/vgg16.keras')
    results.append(evaluate_classifier(
        vgg, create_dataset(parts['test'], preprocess=True), parts['test'], 'VGG16',
    ))
    return pd.DataFrame(results)


def save_yolo_predictions(model, paths, name, size):
    examples = []
    for result in model.predict(
        source=[str(path) for path in paths], stream=True,
        conf=0.25, imgsz=size, device='cpu', verbose=False,
    ):
        if len(examples) < 6:
            examples.append((Path(result.path).name, result.plot()[..., ::-1]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (filename, image) in zip(axes.ravel(), examples):
        ax.imshow(image)
        ax.set_title(filename, fontsize=9)
    fig.tight_layout()
    fig.savefig(data_dir / f'{name}.png', dpi=150)
    plt.close(fig)


def train_yolo(parts):
    # 1.4. Проверка предобученной YOLOv8 на тестовых изображениях.
    weights = data_dir / 'models/yolov8n.pt'
    pretrained = YOLO(str(weights))
    save_yolo_predictions(pretrained, parts['test']['filepath'], 'yolo_pretrained', 640)

    # 1.5-1.7. Дообучение класса cat по подготовленной разметке.
    # Этот класс выбран из-за ошибок исходной YOLO на изображениях кроликов.
    # Изображения no_cats используются как фон без файлов разметки.
    config = str(data_dir / 'dataset/data.yaml')
    trained_weights = data_dir / 'models/yolo_finetuned.pt'
    with tempfile.TemporaryDirectory(prefix='yolo_', dir=data_dir / 'models') as work_dir:
        result = pretrained.train(
            data=config, epochs=epochs, patience=10, imgsz=320, batch=8,
            device='cpu', workers=0, seed=random_state, deterministic=True, amp=False,
            project=work_dir, name='train', plots=False,
        )
        shutil.copy2(Path(result.save_dir) / 'weights/best.pt', trained_weights)

        # 1.8. Проверка дообученной модели на отложенной тестовой выборке.
        trained = YOLO(str(trained_weights))
        metrics = trained.val(data=config, split='test', imgsz=320, batch=8,
                              device='cpu', workers=0, plots=False,
                              project=work_dir, name='test')
    table = pd.DataFrame([{'mAP50': float(metrics.box.map50),
                           'mAP50-95': float(metrics.box.map)}])
    save_yolo_predictions(trained, parts['test']['filepath'], 'yolo_after_finetuning', 320)
    for cache in (data_dir / 'dataset').rglob('*.cache'):
        cache.unlink()
    return table


def save_results(classification, detection):
    text = 'Задание 9. Базовая часть\n\n'
    text += 'Выборки: train — 140, validation — 30, test — 30 изображений.\n'
    text += 'Разметка: 111 кошек на 100 кадрах; 100 кадров без кошек.\n\n'
    text += 'Precision, Recall и F1 рассчитаны для класса «Кошка».\n'
    text += classification.round(4).to_string(index=False) + '\n\n'
    text += 'Дообученная YOLOv8, тестовая выборка:\n'
    text += detection.round(4).to_string(index=False) + '\n\n'
    text += 'Разметка выполнена локально: TXT рядом с изображениями кошек в dataset.\n'
    (data_dir / 'results.txt').write_text(text, encoding='utf-8')
    print(text)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(random_state)
    tf.config.experimental.enable_op_determinism()
    (data_dir / 'models').mkdir(parents=True, exist_ok=True)
    # 1.1. Разделение исходных изображений в соотношении 70/15/15.
    parts = prepare_data()
    classification = train_classifiers(parts)
    detection = train_yolo(parts)
    save_results(classification, detection)


if __name__ == '__main__':
    main()
