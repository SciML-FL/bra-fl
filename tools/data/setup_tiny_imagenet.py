"""Copy Tiny ImageNet validation images into class-specific directories."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def create_val_img_folder(
    dataset_path: str | Path,
    destination_path: str | Path,
    *,
    overwrite_existing: bool = False,
) -> int:
    """Create class folders and copy validation images into them."""
    dataset_path = Path(dataset_path)
    destination_path = Path(destination_path)
    image_dir = dataset_path / "images"
    annotations_path = dataset_path / "val_annotations.txt"

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Validation image directory is missing: {image_dir}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Validation annotations are missing: {annotations_path}")

    copies: list[tuple[Path, Path]] = []
    for line in annotations_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"Malformed annotation line: {line!r}")
        image_name, class_name = fields[:2]
        source = image_dir / image_name
        target = destination_path / class_name / image_name
        if not source.is_file():
            continue
        copies.append((source, target))

    collisions = [target for _, target in copies if target.exists()]
    if collisions and not overwrite_existing:
        preview = ", ".join(str(path) for path in collisions[:3])
        raise FileExistsError(
            f"Refusing to overwrite {len(collisions)} image(s): {preview}. "
            "Pass --overwrite-existing deliberately."
        )

    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return len(copies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("destination_dir", type=Path)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    copied = create_val_img_folder(
        args.validation_dir,
        args.destination_dir,
        overwrite_existing=args.overwrite_existing,
    )
    print(f"Copied {copied} validation image(s) to {args.destination_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
