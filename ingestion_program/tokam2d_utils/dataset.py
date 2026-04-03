from pathlib import Path
from typing import Callable, Optional

import h5py
import torch
import numpy as np
from torchvision.datasets.vision import VisionDataset


from .xml_loader import XMLLoader


class TokamDataset(VisionDataset):
    def __init__(
        self,
        root: Path = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        include_unlabeled: bool = False,  # TODO: implement this
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms
        )
        self.include_unlabeled = include_unlabeled

        self.extract_annotations()
        self.load_images()

    def load_images(self):
        data_files = list(self.root.glob("*.h5"))
        self.images, self.idx_to_frame = [], []
        for file_path in data_files:
            with h5py.File(file_path) as f:
                data = f["density"]
                frame_indices = [
                    f'{file_path.stem}-{idx}' for idx in f["indices"]
                ]
                self.images.append(torch.tensor(
                    np.array(
                        [
                            np.expand_dims(image, axis=0)
                            for idx, image in zip(frame_indices, data)
                            if (
                                self.annotations is None
                                or self.include_unlabeled
                                or idx in self.annotations
                            )
                        ]
                    )
                ))
                self.idx_to_frame.extend([
                    idx for idx in frame_indices
                    if (
                        self.annotations is None
                        or self.include_unlabeled
                        or idx in self.annotations
                    )
                ])
        self.images = torch.cat(self.images, dim=0)
        self.num_frames = self.images.shape[0]
        assert self.num_frames == len(self.idx_to_frame), (
            f"Number of frames mismatch: {self.num_frames} vs "
            f"{len(self.idx_to_frame)}"
        )
        self.image_width = self.images.shape[2]
        self.image_height = self.images.shape[1]
        print(f"Loaded {self.num_frames} frames.")

    def extract_annotations(self):
        self.annotations = {}
        for file_path in self.root.glob("*.xml"):
            file_annotations = XMLLoader(file_path)()
            self.annotations.update(file_annotations)
        if len(self.annotations) == 0:
            self.annotations = None
            print("No label files found")
        else:
            print(
                f"Found annotations for {len(self.annotations)} frames "
                f"in {len(list(self.root.glob('*.xml')))} files."
            )

    def __getitem__(self, index: int):
        frame_index = self.idx_to_frame[index]
        if self.annotations is not None and frame_index in self.annotations:
            boxes = self.annotations[frame_index]
            labels = torch.ones(len(boxes), dtype=torch.int64)
        else:
            boxes = None
            labels = None
        image = self.images[index].float()
        target = {
            "boxes": boxes,
            "labels": labels,
            "frame_index": frame_index,
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return self.num_frames
