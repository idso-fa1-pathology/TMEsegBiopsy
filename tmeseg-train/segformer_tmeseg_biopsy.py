import os
import random as rn
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from transformers import TFAutoModelForSemanticSegmentation


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)


def read_image(img_path, image_size):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return image


def read_mask(mask_path, image_size):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(
        mask,
        [image_size, image_size],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    return mask


def aug_transforms(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.8, 2.0)
    image = tf.image.random_hue(image, 0.1)
    return image


def load_img_train(img_path, mask_path, image_size):
    image = read_image(img_path, image_size)
    mask = read_mask(mask_path, image_size)

    image = aug_transforms(image)

    # For TF SegFormer: channels-first input, C x H x W
    image = tf.transpose(image, (2, 0, 1))

    # Mask should be H x W integer class IDs
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)

    return image, mask


def build_dataset(image_paths, mask_paths, image_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    dataset = dataset.map(
        lambda img, mask: load_img_train(img, mask, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_image_and_mask_paths(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))

    image_names = [Path(path).stem for path in image_paths]

    mask_paths = [
        os.path.join(mask_dir, f"mask_{name}.png")
        for name in image_names
    ]

    return image_paths, mask_paths


def check_mask_paths(mask_paths):
    missing_masks = [path for path in mask_paths if not os.path.exists(path)]

    if missing_masks:
        print(f"[WARNING] Missing {len(missing_masks)} mask files.")
        print("First few missing masks:")
        for path in missing_masks[:10]:
            print(path)

        raise FileNotFoundError("Some mask files are missing.")


def main(
    image_dir,
    mask_dir,
    image_size,
    learning_rate,
    batch_size,
    num_epochs,
    model_checkpoint,
    output_dir,
    output_suffix,
    seed,
    id2label,
    label2id,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    image_paths, mask_paths = get_image_and_mask_paths(
        image_dir=image_dir,
        mask_dir=mask_dir,
    )

    num_img = len(image_paths)
    print(f"Number of training images: {num_img}")

    if num_img == 0:
        raise ValueError(f"No PNG images found in: {image_dir}")

    check_mask_paths(mask_paths)

    train_ds = build_dataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        image_size=image_size,
        batch_size=batch_size,
    )

    optimizer = Adam(learning_rate=learning_rate)

    num_labels = len(id2label)

    model = TFAutoModelForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.compile(optimizer=optimizer)

    model_name = Path(model_checkpoint).name
    model_id = os.path.join(output_dir, f"{model_name}-{output_suffix}")

    steps_per_epoch = max(1, num_img // batch_size)

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        verbose=1,
    )

    model.save_pretrained(model_id)
    print(f"Model saved to: {model_id}")


if __name__ == "__main__":

    # -----------------------
    # Hyperparameter settings
    # -----------------------
    seed = 2023

    image_dir = "/path_to_tme_annotations/image"
    mask_dir = "/path_to_tme_annotations/maskPng"

    image_size = 512
    learning_rate = 0.0001
    batch_size = 8
    num_epochs = 60

    model_checkpoint = "../tmeseg-infer/model/mit-b3-finetuned-TCGAbcss9class"
    output_dir = "../tmeseg-infer/model"
    output_suffix = "biopsy"

    # -----------------------
    # Label settings
    # -----------------------
    id2label = {
        0: "background",
        1: "tumor",
        2: "stroma",
        3: "benign_epi",
        4: "necrosis_hemo",
        5: "adipose",
    }

    label2id = {name: idx for idx, name in id2label.items()}

    main(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        model_checkpoint=model_checkpoint,
        output_dir=output_dir,
        output_suffix=output_suffix,
        seed=seed,
        id2label=id2label,
        label2id=label2id,
    )