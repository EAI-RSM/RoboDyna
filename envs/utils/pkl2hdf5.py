import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def _stack_camera_streams(obs, cam_names):
    """Horizontally stack several cameras' rgb streams into one (N,H,W,3) uint8 array for the
    preview video. Cameras may differ in resolution (head 320x240, countertop 640x480), so each is
    resized to a common EVEN height and the total width is forced even (libx264 yuv420p needs it).
    Returns None if none of the requested cameras are present."""
    import cv2
    streams = [np.asarray(obs[c]["rgb"]) for c in cam_names if c in obs and "rgb" in obs[c]]
    if not streams:
        return None
    if len(streams) == 1:
        return streams[0]
    H = max(s.shape[1] for s in streams)
    H += H % 2
    n = min(s.shape[0] for s in streams)  # align frame counts (should be equal)
    frames = []
    for i in range(n):
        tiles = []
        for s in streams:
            f = s[i]
            h, w = f.shape[:2]
            w2 = int(round(w * (H / h))); w2 += w2 % 2
            tiles.append(cv2.resize(f, (w2, H), interpolation=cv2.INTER_AREA))
        frames.append(np.hstack(tiles))
    arr = np.asarray(frames)
    if arr.shape[2] % 2:
        arr = arr[:, :, :-1, :]
    return arr


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path, fps=30.0):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    # preview video = head (left) + countertop (right) side by side; fall back to whatever single
    # camera exists for embodiments without both.
    obs = data_list["observation"]
    vid = _stack_camera_streams(obs, ["head_camera", "countertop_camera"])
    if vid is None:
        _vid_cam = next((c for c in ("countertop_camera", "head_camera", "front_camera")
                         if c in obs and "rgb" in obs[c]), None)
        vid = np.array(obs[_vid_cam]["rgb"])
    # fps = 250/save_freq -> the preview video plays at REAL sim time (so its length == the actual
    # motion duration); default 30 keeps legacy behavior for callers that don't pass it.
    images_to_video(vid, out_path=video_path, fps=fps)

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path, fps=30.0):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path, fps=fps)
