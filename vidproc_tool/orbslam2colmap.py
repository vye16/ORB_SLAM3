import argparse
import enum
import numpy as np
import os
import yaml

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


class State(enum.Enum):
    SYSTEM_NOT_READY=-1
    NO_IMAGES_YET=0
    NOT_INITIALIZED=1
    OK=2
    RECENTLY_LOST=3
    LOST=4
    OK_KLT=5


def get_tracking_states(state_file):
    """
    return a dict of states for each timestamp
    """
    state_data = np.loadtxt(state_file)
    state_dict = {row[0] : State(row[1]) for row in state_data}
    return state_dict


def get_camera_poses(traj_file):
    """
    return a dict of poses for each timestamp
    """
    # 8 columns, ts, 3 element translation, 4 element quaternion
    traj_data = np.loadtxt(traj_file)
    # timestamp in nanoseconds
    ns = traj_data[:, 0]  # (N)

    # world to cam transform
    T_wc = traj_data[:, 1:8]  # (N, 7), [trans, quat]
    return dict(zip(ns, T_wc))


def interpolate_poses(pose_dict, state_dict):
    all_times = sorted(state_dict.keys())
    ok_times = sorted([t for t, val in state_dict.items() if val ==  State.OK])
    assert(all(t in pose_dict for t in ok_times))
    ok_poses = [pose_dict[t] for t in ok_times]

    # fill endpoints
    min_time, min_ok_time = min(all_times), min(ok_times)
    if min_time < min_ok_time:
        ok_times.insert(0, min_time)
        ok_poses.insert(0, pose_dict[min_ok_time])
    max_time, max_ok_time = max(state_dict.keys()), max(ok_times)
    if max_time > max_ok_time:
        ok_times.append(max_time)
        ok_poses.append(pose_dict[max_ok_time])

    print("interpolating {} poses with {} points".format(len(all_times), len(ok_times)))
    assert(len(ok_times) == len(ok_poses))

    # interpolate with slerp on rots and lerp on trans
    ok_poses = np.array(ok_poses)  # (N, 4, 4)
    ok_trans = ok_poses[:, :3]  # (N, 3)
    ok_rots = Rotation.from_quat(ok_poses[:, 3:])  # (N, 4)

    lerp = interp1d(ok_times, ok_trans, axis=0)
    slerp = Slerp(ok_times, ok_rots)
    trans_interp = lerp(all_times)
    rots_interp = slerp(all_times)
    return all_times, rots_interp, trans_interp


def write_colmap_images(img_dir, out_dir, rots_wc, trans_wc):
    """
    :param img_dir
    :param out_dir
    :param rots_wc (N-len list of Rotations)
    :param trans_wc (N, 3)
    """
    img_files = sorted(os.listdir(img_dir))
    out_file = os.path.join(out_dir, "images.txt")
    quats = rots_wc.as_quat()  # (N, 4), xyzw format
    quats = np.concatenate([quats[:, 0:1], quats[:, 1:]], axis=1)  # wxyz format
    with open(out_file, "w") as f:
        for i, img in enumerate(img_files):
            quat_str = "{} {} {} {}".format(*quats[i])
            trans_str = "{} {} {}".format(*trans_wc[i])
            print(i, img, quat_str, trans_str)
            f.write("{} {} {} 1 {}\n\n".format(i+1, quat_str, trans_str, img))
    print("wrote camera poses to {}".format(out_file))
    

def write_colmap_points(out_dir):
    # write an empty file for the points
    out_file = os.path.join(out_dir, "points3D.txt")
    with open(out_file, "w") as f:
        pass
    print("wrote empty file to {}".format(out_file))


def write_colmap_camera(camera_file, out_dir):
    with open(camera_file, "r") as f:
        f.readline()
        params = yaml.safe_load(f)

    w, h, fx, fy, cx, cy = (
        params["Camera.width"],
        params["Camera.height"],
        params["Camera.fx"],
        params["Camera.fy"],
        params["Camera.cx"],
        params["Camera.cy"],
    )

    out_file = os.path.join(out_dir, "cameras.txt")
    with open(out_file, "w") as f:
        f.write("{} PINHOLE {} {} {} {} {} {}".format(1, w, h, fx, fy, int(cx), int(cy)))

    print("Wrote camera to {}".format(out_file))


def main(img_dir, out_dir, state_file, traj_file, camera_file):
    os.makedirs(out_dir, exist_ok=True)
    n_imgs = len(os.listdir(img_dir))

    pose_dict = get_camera_poses(traj_file)
    state_dict = get_tracking_states(state_file)

    times, rots_wc, trans_wc = interpolate_poses(pose_dict, state_dict)
    if len(times) != n_imgs:
        print("Expected {} times, found {}, exiting".format(n_imgs, len(times)))
        return

    write_colmap_images(img_dir, out_dir, rots_wc, trans_wc)
    write_colmap_camera(camera_file, out_dir)
    write_colmap_points(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        required=True,
        type=str,
        help="Image dir",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Where to save the output poses",
    )
    parser.add_argument(
        "--traj",
        required=True,
        type=str,
        help="camera trajectory output of ORBSLAM",
    )
    parser.add_argument(
        "--states",
        required=True,
        type=str,
        help="state output of ORBSLAM",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="480p.yaml",
        help="settings file used to run ORBSLAM",
    )
    args = parser.parse_args()

    print("post processing with args {}".format(args))
    main(args.img_dir, args.out_dir, args.states, args.traj, args.settings)
