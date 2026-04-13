import argparse
import json
import logging
import random
import ssl
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

import albumentations as A
import cv2
import yaml

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO2017_TRAIN_URL = "https://images.cocodataset.org/zips/train2017.zip"
COCO2017_VAL_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO2017_ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
EPS = 1e-6


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare COCO datasets in YOLO format.")
    parser.add_argument(
        "--project-root",
        type=Path,
        nargs="?",
        const=Path("."),
        default=Path(__file__).resolve().parents[1],
        help="Project root path. If the flag is provided without a value, current directory is used.",
    )
    parser.add_argument("--dataset", type=str, default="coco128", choices=["coco2017", "coco128"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-copies", type=int, default=0, help="Offline augmented copies per train image")
    return parser.parse_args()


def download_and_extract(url: str, zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if any(extract_dir.iterdir()):
        logging.info("Using existing extracted data at %s", extract_dir)
        return

    logging.info("Downloading %s", url)
    ssl_context = ssl._create_unverified_context()
    with urlopen(url, context=ssl_context) as response, zip_path.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)
    logging.info("Extracting to %s", extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)


def download_file(url: str, zip_path: Path) -> None:
    ssl_context = ssl._create_unverified_context()
    with urlopen(url, context=ssl_context) as response, zip_path.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve any of the expected dataset paths")


def reset_generated_dirs(project_root: Path) -> None:
    for relative_path in ["data/images/train", "data/images/val", "data/labels/train", "data/labels/val"]:
        target_dir = project_root / relative_path
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)


def download_coco128(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "coco128.zip"
    extract_dir = target_dir / "coco128_raw"

    if not extract_dir.exists():
        logging.info("Downloading COCO128 from %s", COCO128_URL)
        download_file(COCO128_URL, zip_path)
        logging.info("Extracting dataset to %s", extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        logging.info("Using existing extracted dataset at %s", extract_dir)

    return extract_dir / "coco128"


def download_coco2017(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = target_dir / "coco2017_raw"
    train_zip = target_dir / "train2017.zip"
    val_zip = target_dir / "val2017.zip"
    ann_zip = target_dir / "annotations_trainval2017.zip"

    train_extract = raw_dir / "train2017_archive"
    val_extract = raw_dir / "val2017_archive"
    ann_extract = raw_dir / "annotations_archive"

    download_and_extract(COCO2017_TRAIN_URL, train_zip, train_extract)
    download_and_extract(COCO2017_VAL_URL, val_zip, val_extract)
    download_and_extract(COCO2017_ANN_URL, ann_zip, ann_extract)

    return raw_dir


def load_names(coco_root: Path) -> List[str]:
    candidates = [
        coco_root / "coco128.yaml",
        coco_root.parent / "coco128.yaml",
        coco_root.parent.parent / "coco128.yaml",
    ]

    for yaml_path in candidates:
        if not yaml_path.exists():
            continue

        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names", []) if isinstance(data, dict) else []
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        if isinstance(names, list) and names:
            logging.info("Loaded %d class names from %s", len(names), yaml_path)
            return names

    logging.warning("Could not find class names from coco128.yaml, falling back to 80 COCO placeholders")
    return [f"class_{i}" for i in range(80)]


def load_coco_categories(annotation_path: Path) -> List[Dict[str, object]]:
    with annotation_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    categories = data.get("categories", []) if isinstance(data, dict) else []
    if not categories:
        raise RuntimeError(f"No categories found in {annotation_path}")

    return sorted(categories, key=lambda item: int(item["id"]))


def yolo_txt_to_bboxes(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    bboxes: List[List[float]] = []
    class_labels: List[int] = []
    if not label_path.exists():
        return bboxes, class_labels

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            class_labels.append(cls_id)
            bboxes.append([x, y, w, h])

    return sanitize_yolo_bboxes(bboxes, class_labels)


def sanitize_yolo_bboxes(
    bboxes: List[List[float]],
    class_labels: List[int],
    eps: float = EPS,
) -> Tuple[List[List[float]], List[int]]:
    """Clamp YOLO normalized bboxes to [0, 1] and drop invalid boxes."""
    clean_boxes: List[List[float]] = []
    clean_labels: List[int] = []

    for bbox, cls_id in zip(bboxes, class_labels):
        if len(bbox) != 4:
            continue

        x, y, w, h = [float(v) for v in bbox]

        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0

        x1 = min(1.0, max(0.0, x1))
        y1 = min(1.0, max(0.0, y1))
        x2 = min(1.0, max(0.0, x2))
        y2 = min(1.0, max(0.0, y2))

        new_w = x2 - x1
        new_h = y2 - y1
        if new_w <= eps or new_h <= eps:
            continue

        new_x = (x1 + x2) / 2.0
        new_y = (y1 + y2) / 2.0

        clean_boxes.append([new_x, new_y, new_w, new_h])
        clean_labels.append(int(cls_id))

    return clean_boxes, clean_labels


def save_yolo_labels(label_path: Path, bboxes: List[List[float]], class_labels: List[int]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, bbox in zip(class_labels, bboxes):
            x, y, w, h = bbox
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def coco_bbox_to_yolo(bbox: List[float], image_width: int, image_height: int) -> List[float] | None:
    x, y, w, h = [float(value) for value in bbox]
    if w <= EPS or h <= EPS:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + w)
    y2 = min(float(image_height), y + h)

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= EPS or clipped_h <= EPS:
        return None

    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    norm_w = clipped_w / image_width
    norm_h = clipped_h / image_height
    return [center_x, center_y, norm_w, norm_h]


def write_coco_split(
    image_dir: Path,
    annotation_path: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    category_id_to_index: Dict[int, int],
) -> int:
    with annotation_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_lookup = {int(image["id"]): image for image in images}
    annotations_by_image: Dict[int, List[dict]] = {}

    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) == 1:
            continue
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    written = 0
    for image_id, image_info in image_lookup.items():
        source_image = image_dir / image_info["file_name"]
        if not source_image.exists():
            logging.warning("Missing source image: %s", source_image)
            continue

        target_image = output_image_dir / image_info["file_name"]
        shutil.copy2(source_image, target_image)

        labels: List[List[float]] = []
        class_labels: List[int] = []
        for annotation in annotations_by_image.get(image_id, []):
            bbox = coco_bbox_to_yolo(annotation.get("bbox", []), int(image_info["width"]), int(image_info["height"]))
            if bbox is None:
                continue

            category_id = int(annotation["category_id"])
            if category_id not in category_id_to_index:
                continue

            class_labels.append(category_id_to_index[category_id])
            labels.append(bbox)

        save_yolo_labels(output_label_dir / f"{Path(image_info['file_name']).stem}.txt", labels, class_labels)
        written += 1

    return written


def write_sanitized_label_file(src_label: Path, dst_label: Path) -> None:
    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return

    bboxes, class_labels = yolo_txt_to_bboxes(src_label)
    save_yolo_labels(dst_label, bboxes, class_labels)


def build_augmenter() -> A.Compose:
    # Albumentations 2.x expects `size=(h, w)` while older versions accepted height/width.
    try:
        rrc = A.RandomResizedCrop(size=(640, 640), scale=(0.7, 1.0), p=0.3)
    except TypeError:
        rrc = A.RandomResizedCrop(height=640, width=640, scale=(0.7, 1.0), p=0.3)

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            rrc,
            A.GaussNoise(p=0.15),
            A.MotionBlur(blur_limit=5, p=0.1),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.05),
    )


def write_dataset_yaml(project_root: Path, names: List[str]) -> Path:
    dataset_yaml = project_root / "data" / "dataset.yaml"
    payload = {
        "path": str((project_root / "data").resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    with dataset_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    logging.info("Wrote dataset.yaml to %s", dataset_yaml)
    logging.info("Use training config mosaic=0.7 and mixup=0.05 for online augmentation.")
    return dataset_yaml


def prepare_coco2017_dataset(project_root: Path) -> None:
    coco_root = download_coco2017(project_root / "data")
    train_image_dir = first_existing_path(coco_root / "train2017_archive" / "train2017", coco_root / "train2017_archive")
    val_image_dir = first_existing_path(coco_root / "val2017_archive" / "val2017", coco_root / "val2017_archive")
    annotation_dir = first_existing_path(coco_root / "annotations_archive" / "annotations", coco_root / "annotations_archive")
    train_annotation = annotation_dir / "instances_train2017.json"
    val_annotation = annotation_dir / "instances_val2017.json"

    categories = load_coco_categories(train_annotation)
    names = [str(category["name"]) for category in categories]
    category_id_to_index = {int(category["id"]): index for index, category in enumerate(categories)}

    reset_generated_dirs(project_root)

    train_count = write_coco_split(
        image_dir=train_image_dir,
        annotation_path=train_annotation,
        output_image_dir=project_root / "data" / "images" / "train",
        output_label_dir=project_root / "data" / "labels" / "train",
        category_id_to_index=category_id_to_index,
    )
    val_count = write_coco_split(
        image_dir=val_image_dir,
        annotation_path=val_annotation,
        output_image_dir=project_root / "data" / "images" / "val",
        output_label_dir=project_root / "data" / "labels" / "val",
        category_id_to_index=category_id_to_index,
    )

    write_dataset_yaml(project_root, names)
    logging.info("Prepared full COCO2017 dataset | train images: %d | val images: %d", train_count, val_count)


def prepare_split(
    coco_root: Path,
    project_root: Path,
    val_ratio: float,
    seed: int,
    aug_copies: int,
) -> None:
    src_img_dir = coco_root / "images" / "train2017"
    src_lbl_dir = coco_root / "labels" / "train2017"

    train_img_dir = project_root / "data" / "images" / "train"
    val_img_dir = project_root / "data" / "images" / "val"
    train_lbl_dir = project_root / "data" / "labels" / "train"
    val_lbl_dir = project_root / "data" / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in src_img_dir.glob("*.jpg")])
    if not images:
        raise RuntimeError(f"No images found at {src_img_dir}")

    random.seed(seed)
    random.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio))
    val_set = set(images[:val_count])

    augmenter = build_augmenter()
    train_count = 0
    val_count_written = 0

    for img_path in images:
        stem = img_path.stem
        label_path = src_lbl_dir / f"{stem}.txt"

        if img_path in val_set:
            shutil.copy2(img_path, val_img_dir / img_path.name)
            write_sanitized_label_file(label_path, val_lbl_dir / label_path.name)
            val_count_written += 1
            continue

        shutil.copy2(img_path, train_img_dir / img_path.name)
        write_sanitized_label_file(label_path, train_lbl_dir / label_path.name)

        bboxes, class_labels = yolo_txt_to_bboxes(label_path)
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logging.warning("Skipping unreadable image: %s", img_path)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        for i in range(aug_copies):
            try:
                transformed = augmenter(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            except ValueError as exc:
                logging.warning("Skipping augmentation for %s due to bbox error: %s", img_path.name, exc)
                continue

            aug_bboxes, aug_labels = sanitize_yolo_bboxes(
                transformed["bboxes"], transformed["class_labels"]
            )
            aug_rgb = transformed["image"]
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            aug_name = f"{stem}_aug{i}.jpg"
            aug_label = f"{stem}_aug{i}.txt"

            cv2.imwrite(str(train_img_dir / aug_name), aug_bgr)
            save_yolo_labels(train_lbl_dir / aug_label, aug_bboxes, aug_labels)

        train_count += 1

    logging.info("Prepared train images: %d (+augmented copies)", train_count)
    logging.info("Prepared val images: %d", val_count_written)


def main() -> None:
    setup_logging()
    args = parse_args()

    project_root = args.project_root.resolve()
    if args.dataset == "coco2017":
        prepare_coco2017_dataset(project_root)
    else:
        coco_root = download_coco128(project_root / "data")
        names = load_names(coco_root)

        reset_generated_dirs(project_root)
        prepare_split(
            coco_root=coco_root,
            project_root=project_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
            aug_copies=args.aug_copies,
        )
        write_dataset_yaml(project_root, names)

    logging.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
