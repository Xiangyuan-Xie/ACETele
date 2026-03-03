import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import rerun as rr


class Hdf5RerunViewer:
    def __init__(
        self,
        h5_path,
        save_rrd=None,
        stride=1,
        max_frames=None,
        image_shape=None,
        image_dtype=None,
    ):
        self.h5_path = Path(h5_path)
        self.save_rrd = save_rrd
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self.image_shape = image_shape
        self.image_dtype = image_dtype

    def run(self):
        if not self.h5_path.exists():
            raise SystemExit(f"HDF5 not found: {self.h5_path}")

        self._init_rerun()
        with h5py.File(self.h5_path, "r") as f:
            datasets = self._collect_datasets(f)
            image_paths = sorted([p for p in datasets if "images" in p])
            compress_len = datasets.get("compress_len")
            shape_override = self._parse_shape(self.image_shape)
            frame_count = self._get_frame_count(datasets)
            if frame_count is None:
                return
            max_frames = self.max_frames or frame_count
            for frame_idx in range(0, min(frame_count, max_frames), self.stride):
                rr.set_time("frame", sequence=frame_idx)
                for path, ds in datasets.items():
                    if path in image_paths:
                        col_idx = image_paths.index(path)
                        length = ds.shape[1]
                        if compress_len is not None and compress_len.ndim == 2:
                            if col_idx < compress_len.shape[1]:
                                length = int(compress_len[frame_idx, col_idx])
                        bytes_row = bytes(ds[frame_idx][:length])
                        self._log_image(path, bytes_row, shape_override, self.image_dtype)
                    elif path == "compress_len":
                        self._log_numeric(path, ds, frame_idx)
                    else:
                        self._log_numeric(path, ds, frame_idx)

    def _init_rerun(self):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if self.save_rrd or not has_display:
            output_rrd = self.save_rrd or str(self.h5_path.with_suffix(".rrd"))
            rr.init("hdf5_viewer", spawn=False)
            rr.save(output_rrd)
        else:
            rr.init("hdf5_viewer", spawn=True)

    def _collect_datasets(self, h5_file):
        datasets = {}

        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                datasets[name] = node

        h5_file.visititems(visitor)
        return datasets

    def _get_frame_count(self, datasets):
        for ds in datasets.values():
            if ds.ndim >= 1:
                return ds.shape[0]
        return None

    def _parse_shape(self, value):
        if value is None:
            return None
        parts = [int(p) for p in value.lower().split("x") if p.strip()]
        if not parts:
            return None
        if len(parts) == 2:
            return (parts[0], parts[1], 1)
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
        return None

    def _guess_shape(self, byte_len, dtype_hint=None):
        common_sizes = [
            (848, 480),
            (848, 240),
            (640, 480),
            (640, 360),
            (424, 240),
            (1280, 720),
        ]
        if dtype_hint == "uint16":
            if byte_len % 2 == 0:
                px = byte_len // 2
                for h, w in common_sizes:
                    if h * w == px:
                        return (h, w, 1), np.uint16
            return None, np.uint16
        if dtype_hint == "uint8":
            for h, w in common_sizes:
                if h * w == byte_len:
                    return (h, w, 1), np.uint8
            return None, np.uint8
        for h, w in common_sizes:
            if h * w == byte_len:
                return (h, w, 1), np.uint8
        if byte_len % 2 == 0:
            px = byte_len // 2
            for h, w in common_sizes:
                if h * w == px:
                    return (h, w, 1), np.uint16
        return None, np.uint8

    def _log_numeric(self, path, data, frame_idx):
        if data.ndim == 1:
            rr.log(path, rr.Scalars([float(data[frame_idx])]))
            return
        if data.ndim == 2:
            for i in range(data.shape[1]):
                rr.log(f"{path}/{i}", rr.Scalars([float(data[frame_idx, i])]))
            return
        rr.log(path, rr.Tensor(data[frame_idx]))

    def _log_image(self, path, bytes_row, shape_override, dtype_hint):
        byte_len = len(bytes_row)
        if byte_len == 0:
            return
        if shape_override:
            shape = shape_override
            dtype = np.uint16 if dtype_hint == "uint16" else np.uint8
        else:
            shape, dtype = self._guess_shape(byte_len, dtype_hint)
        if shape:
            arr = np.frombuffer(bytes_row, dtype=dtype)
            if arr.size != shape[0] * shape[1] * shape[2]:
                rr.log(path, rr.Tensor(arr))
                return
            arr = arr.reshape(shape)
            if shape[2] == 1:
                arr = arr.reshape(shape[0], shape[1])
            rr.log(path, rr.Image(arr))
            return
        rr.log(path, rr.Tensor(np.frombuffer(bytes_row, dtype=np.uint8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path")
    parser.add_argument("--save-rrd", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-shape", default=None)
    parser.add_argument("--image-dtype", choices=["uint8", "uint16"], default=None)
    args = parser.parse_args()

    Hdf5RerunViewer(
        args.h5_path,
        save_rrd=args.save_rrd,
        stride=args.stride,
        max_frames=args.max_frames,
        image_shape=args.image_shape,
        image_dtype=args.image_dtype,
    ).run()


if __name__ == "__main__":
    main()
