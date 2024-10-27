import os
import warnings

import numpy as np
from numpy import pi

warnings.filterwarnings("ignore")
import random

from tqdm import tqdm

seed = 7607
BATCH_SIZE = 1000
MAX_CONTRAST = 1.75
MIN_CONTRAST = 0.01 * MAX_CONTRAST

MIN_SIZE = 4
MAX_SIZE = 25

num_contrasts = 6
contrast_increment = (MAX_CONTRAST / MIN_CONTRAST) ** (1 / (num_contrasts - 1))

num_sizes = 8
size_increment = (MAX_SIZE / MIN_SIZE) ** (1 / (num_sizes - 1))

PATH_TO_SAVE = "../../data/gabors_gaussian/"


class GaborSet:
    def __init__(
        self,
        canvas_size,  # width x height
        center_range,  # [x_start, x_end, y_start, y_end]
        sizes,  # +/- 2 SD of Gaussian envelope
        spatial_frequencies,  # cycles / 4 SD of envelope (i.e. depends on size)
        contrasts,
        orientations,
        phases,
        relative_sf=True,  # scale spatial frequency by size
    ):
        self.canvas_size = canvas_size
        cr = center_range
        self.locations = np.array(
            [[x, y] for x in range(cr[0], cr[1]) for y in range(cr[2], cr[3])]
        )
        self.sizes = sizes
        self.spatial_frequencies = spatial_frequencies
        self.contrasts = contrasts
        if type(orientations) is not list:
            self.orientations = np.arange(orientations) * pi / orientations
        else:
            self.orientations = orientations
        if type(phases) is not list:
            self.phases = np.arange(phases) * (2 * pi) / phases
        else:
            self.phases = phases
        self.num_params = [
            self.locations.shape[0],
            len(sizes),
            len(spatial_frequencies),
            len(contrasts),
            len(self.orientations),
            len(self.phases),
        ]
        self.relative_sf = relative_sf

    def params_from_idx(self, idx):
        c = np.unravel_index(idx, self.num_params)
        location = self.locations[c[0]]
        size = self.sizes[c[1]]
        spatial_frequency = self.spatial_frequencies[c[2]]
        if self.relative_sf:
            spatial_frequency /= size
        contrast = self.contrasts[c[3]]
        orientation = self.orientations[c[4]]
        phase = self.phases[c[5]]
        return location, size, spatial_frequency, contrast, orientation, phase

    def params_dict_from_idx(self, idx):
        (
            location,
            size,
            spatial_frequency,
            contrast,
            orientation,
            phase,
        ) = self.params_from_idx(idx)
        return {
            "location": location,
            "size": size,
            "spatial_frequency": spatial_frequency,
            "contrast": contrast,
            "orientation": orientation,
            "phase": phase,
        }

    def gabor_from_idx(self, idx):
        return self.gabor(*self.params_from_idx(idx))

    def gabor(self, location, size, spatial_frequency, contrast, orientation, phase):
        x, y = np.meshgrid(
            np.arange(self.canvas_size[0]) - location[0],
            np.arange(self.canvas_size[1]) - location[1],
        )
        R = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )
        coords = np.stack([x.flatten(), y.flatten()])
        x, y = R.dot(coords).reshape((2,) + x.shape)
        envelope = contrast * np.exp(-(x**2 + y**2) / (2 * (size / 4) ** 2))

        grating = np.cos(spatial_frequency * x * (2 * pi) + phase)
        return envelope * grating

    def image_batches(self, batch_size):
        num_stims = np.prod(self.num_params)
        for batch_start in np.arange(0, num_stims, batch_size):
            batch_end = np.minimum(batch_start + batch_size, num_stims)
            images = [self.gabor_from_idx(i) for i in range(batch_start, batch_end)]
            yield np.array(images)

    def images(self):
        num_stims = np.prod(self.num_params)
        return np.array([self.gabor_from_idx(i) for i in range(num_stims)])


if __name__ == "__main__":
    # pre-saving gabors speeds up the search of the optimal one
    random.seed(seed)
    np.random.seed(seed)

    canvas_size = [MAX_SIZE, MAX_SIZE]
    # value from here https://github.com/ecker-lab/burg2021_learning_divisive_normalization/blob/main/divisivenormalization/insilico.py#L211C38-L211C58
    x_start = 0
    x_end = MAX_SIZE
    y_start = 0
    y_end = MAX_SIZE
    min_size = MIN_SIZE

    min_sf = 1.3**-1
    num_sf = 10
    sf_increment = 1.3
    min_contrast = MIN_CONTRAST

    num_orientations = 12
    num_phases = 8

    center_range = [x_start, x_end, y_start, y_end]
    sizes = min_size * size_increment ** np.arange(num_sizes)
    sfs = min_sf * sf_increment ** np.arange(num_sf)
    c = min_contrast * contrast_increment ** np.arange(num_contrasts)
    g = GaborSet(
        canvas_size,
        center_range,
        sizes,
        sfs,
        c,
        num_orientations,
        num_phases,
    )

    num_stims = np.prod(g.num_params)
    print(num_stims)
    for batch_idx, batch_start in tqdm(
        enumerate(np.arange(0, num_stims, BATCH_SIZE)), total=num_stims / BATCH_SIZE
    ):
        batch_end = np.minimum(batch_start + BATCH_SIZE, num_stims)
        images = [g.gabor_from_idx(i) for i in range(batch_start, batch_end)]
        images_or = np.array(images)
        np.save(f"{PATH_TO_SAVE}{batch_idx}.npy", images_or)
