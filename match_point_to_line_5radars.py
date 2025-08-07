# --- LBFGS optimization function ---
def optimize_with_lbfgs(
    r, t, matched_pairs, opt_step, opt_lr, r_reg_weight, t_reg_weight=0.0
):
    """
    Optimize r, t using LBFGS optimizer. Returns histories for plotting.
    r_reg_weight: regularization weight for r
    t_reg_weight: regularization weight for t (default 0)
    """
    device = r.device
    loss_history = []
    main_loss_history = []
    reg_loss_history = []
    r_hist = []
    t_hist = []

    params = [r, t]
    optimizer = torch.optim.LBFGS(
        params,
        lr=opt_lr,
        max_iter=opt_step,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        main_loss = calculate_point_to_line_loss(r, t, matched_pairs)
        reg_loss = r_reg_weight * r.norm() + t_reg_weight * t.norm()
        loss = main_loss + reg_loss
        loss.backward()
        # Record histories for each step
        loss_history.append(loss.item())
        main_loss_history.append(main_loss.item())
        reg_loss_history.append(reg_loss.item())
        r_hist.append(r.detach().cpu().numpy().tolist())
        t_hist.append(t.detach().cpu().numpy().tolist())
        return loss

    print("Starting LBFGS optimization...")
    optimizer.step(closure)
    print("LBFGS optimization finished.")

    return {
        "loss_history": loss_history,
        "main_loss_history": main_loss_history,
        "reg_loss_history": reg_loss_history,
        "r_hist": r_hist,
        "t_hist": t_hist,
        "r": r,
        "t": t,
    }


"""
this file takes seq_id and frame_id  and optimization params (step,lr) from CLI
then loads the all 6 radar channels: calibrations, filename
then use Adam optimizer to optimize 5 RTs (for radar 1-5, 6 parameters each) to simultaneously minimize point-to-line distance
"""

line_json_root = "/home/palakons/linear_hough_man_radar/public/json"

import argparse
import torch
from tqdm import trange, tqdm
from pytorch3d.transforms import axis_angle_to_matrix
import pandas as pd
import json, os, time
import numpy as np


from torchviz import make_dot
import matplotlib.pyplot as plt


def point_to_line_distance(points, line_p0, line_p1):
    """
    Calculate the distance from points to a line defined by two points (line_p0, line_p1).
    :param points: Tensor of shape (N, 3) where N is the number of points.
    :param line_p0: Tensor of shape (3,) representing the first point of the line.
    :param line_p1: Tensor of shape (3,) representing the second point of the line.
    :return: Tensor of distances from each point to the line.
    """
    time0 = time.time()
    line_vector = line_p1 - line_p0
    time01 = time.time()
    # print("lineP0:", line_p0, "type:", line_p0.dtype, "device:", line_p0.device)
    # print(
    #     "lineP1:", line_p1, "type:", line_p1.dtype, "device:", line_p1.device
    # )  # Debugging line
    # print(
    #     "line_vector:",
    #     line_vector,
    #     "type:",
    #     line_vector.dtype,
    #     "device:",
    #     line_vector.device,
    # )
    # lineP0: tensor([ 51.1185, -13.9699,  -0.8038], device='cuda:0') type: torch.float32 device: cuda:0
    # lineP1: tensor([ 52.1181, -13.9881,  -0.8228], device='cuda:0') type: torch.float32 device: cuda:0
    # line_vector: tensor([ 0.9996, -0.0182, -0.0190], device='cuda:0') type: torch.float32 device: cuda:0
    line_length = torch.norm(line_vector)
    time02 = time.time()
    if line_length == 0:
        raise ValueError(
            f"line_p0 and line_p1 cannot be the same point: {line_p0}, {line_p1}"
        )
    time03 = time.time()
    # Normalize the line vector
    line_vector_normalized = line_vector / line_length
    time04 = time.time()

    # Ensure line_p0 is shape (3,) and broadcastable
    # print(
    #     f"line_p0 shape: {line_p0.shape}, points shape: {points.shape}"
    # )  # line_p0 shape: torch.Size([3, 3]), points shape: torch.Size([64, 3])
    if line_p0.dim() == 1:
        point_vectors = points - line_p0.unsqueeze(0)
    else:
        point_vectors = points - line_p0
    time1 = time.time()

    # Project point vectors onto the normalized line vector
    projection_length = torch.sum(point_vectors * line_vector_normalized, dim=1)
    time2 = time.time()
    projection = projection_length[:, None] * line_vector_normalized
    time3 = time.time()

    # Calculate the distance from points to the projection on the line
    distances = torch.norm(point_vectors - projection, dim=1)
    time4 = time.time()
    total_time = time4 - time0
    # print(
    #     f"Total time for point_to_line_distance: {time4 - time0:.4f}s, Percent of time spent in each step: "
    #     f"point_vector: {(time1 - time0) / total_time:.2%}, projection_length: {(time2 - time1) / total_time:.2%}, projection: {(time3 - time2) / total_time:.2%}, distance: {(time4 - time3) / total_time:.2%}"
    #     f"\ndetail time percent: line_vector: {(time01 - time0) / total_time:.2%}, line_length: {(time02 - time01) / total_time:.2%}, normalize: {(time03 - time02) / total_time:.2%},  divide: {(time04 - time03) / total_time:.2%} if: { (time1 - time03) / total_time:.2%}"
    # )
    # Total time for point_to_line_distance: 0.0040s, Percent of time spent in each step: point_vector: 95.76%, projection_length: 1.64%, projection: 1.12%, distance: 1.48%
    # detail time percent: line_vector: 0.40%, line_length: 91.69%, normalize: 1.84%,  divide: 0.65% if: 1.83%

    # Total time for point_to_line_distance: 0.0005s, Percent of time spent in each step: point_vector: 65.10%, projection_length: 12.05%, projection: 10.10%, distance: 12.74%
    # detail time percent: line_vector: 4.20%, line_length: 21.33%, normalize: 27.23%,  divide: 5.76% if: 12.35%

    return distances


def calculate_point_to_line_loss_tower(r, t, matched_pairs):
    import time

    device = r.device
    loss = torch.tensor(0.0, device=device)
    for j, pair in enumerate(matched_pairs):
        # for radar1, radar2 in zip([pair["radar1_data"], pair["radar2_data"]]):
        t0 = time.time()
        # Extract radar data
        radar1_data = pair["radar1_data"]
        radar2_data = pair["radar2_data"]
        r1_org = radar1_data["rotation"]
        r2_org = radar2_data["rotation"]
        t1_org = radar1_data["translation"]
        t2_org = radar2_data["translation"]
        radar1_inliers = pair["radar1_inliers"]
        radar2_inliers = pair["radar2_inliers"]
        radar1_rev_idx = pair["radar1_revIdx"]
        radar2_rev_idx = pair["radar2_revIdx"]
        t1 = (
            t[radar1_rev_idx - 1]
            if radar1_rev_idx > 0
            else torch.zeros(3, device=device)
        )
        t2 = (
            t[radar2_rev_idx - 1]
            if radar2_rev_idx > 0
            else torch.zeros(3, device=device)
        )
        r1 = (
            axis_angle_to_matrix(r[radar1_rev_idx - 1])
            if radar1_rev_idx > 0
            else torch.eye(3, device=device)
        )
        r2 = (
            axis_angle_to_matrix(r[radar2_rev_idx - 1])
            if radar2_rev_idx > 0
            else torch.eye(3, device=device)
        )
        t1_data = time.time()
        # Transformations
        inliers1_world = (r1_org @ radar1_inliers["inlierCloud"].T + t1_org[:, None]).T
        lineP0_1_world = (r1_org @ radar1_inliers["lineP0"] + t1_org).T
        lineP1_1_world = (r1_org @ radar1_inliers["lineP1"] + t1_org).T
        inliers2_world = (r2_org @ radar2_inliers["inlierCloud"].T + t2_org[:, None]).T
        lineP0_2_world = (r2_org @ radar2_inliers["lineP0"] + t2_org).T
        lineP1_2_world = (r2_org @ radar2_inliers["lineP1"] + t2_org).T
        inliers1_world_adjusted = (r1 @ inliers1_world.T + t1[:, None]).T
        lineP0_1_world_adjusted = r1 @ lineP0_1_world + t1
        lineP1_1_world_adjusted = r1 @ lineP1_1_world + t1
        inliers2_world_adjusted = (r2 @ inliers2_world.T + t2[:, None]).T
        lineP0_2_world_adjusted = r2 @ lineP0_2_world + t2
        lineP1_2_world_adjusted = r2 @ lineP1_2_world + t2
        t2_data = time.time()
        # Distance calculations
        d1 = point_to_line_distance(
            inliers1_world_adjusted,
            lineP0_2_world_adjusted,
            lineP1_2_world_adjusted,
        ).mean()
        d2 = point_to_line_distance(
            inliers2_world_adjusted,
            lineP0_1_world_adjusted,
            lineP1_1_world_adjusted,
        ).mean()
        t3_data = time.time()
        loss += d1 + d2

        if True:
            # use torchviz
            params = {
                "r": r,
                "t": t,
                "d1": d1,
                "d2": d2,
                "loss": loss,
                "r1": r1,
                "r2": r2,
                "t1": t1,
                "t2": t2,
                "inliers2_world_adjusted": inliers2_world_adjusted,
                "lineP0_2_world_adjusted": lineP0_2_world_adjusted,
                "lineP1_2_world_adjusted": lineP1_2_world_adjusted,
                "inliers1_world_adjusted": inliers1_world_adjusted,
                "lineP0_1_world_adjusted": lineP0_1_world_adjusted,
                "lineP1_1_world_adjusted": lineP1_1_world_adjusted,
                # "line_vector": line_vector,
                # "line_vector_normalized": line_vector_normalized,
            }
            dot = make_dot(loss, params=params, show_attrs=True, max_attr_chars=50)
            dot.render(
                f"/home/palakons/logs/r-r_matching/graph/graph_tower_{j:06d}",
                format="png",
            )
        # print(
        #     f"[Timing] Data extraction: {t1_data-t0:.4f}s, Transformations: {t2_data-t1_data:.4f}s, Distance: {t3_data-t2_data:.4f}s"
        # )  # [Timing] Data extraction: 0.0009s, Transformations: 0.0003s, Distance: 0.0003s

    return loss / len(matched_pairs)


def transform_points(points, rotation, translation):
    """
    Transform points using rotation and translation.
    :param points: Tensor of shape (N, 3) where N is the number of points.
    :param rotation: Tensor of shape (3, 3) representing the rotation matrix.
    :param translation: Tensor of shape (3,) representing the translation vector.
    :return: Transformed points as a tensor of shape (N, 3).
    """
    assert rotation.shape == (
        3,
        3,
    ), f"Rotation must be a 3x3 matrix, got {rotation.shape}"
    assert translation.shape == (
        3,
    ), f"Translation must be a 3-vector, got {translation.shape}"
    assert points.shape[1] == 3, f"Points must be of shape (N, 3), got {points.shape}"
    output = (rotation @ points.T).T + translation
    assert (
        output.shape == points.shape
    ), f"Output shape mismatch: expected {points.shape}, got {output.shape}"
    return output


def preprocess_matched_pairs(matched_pairs, device):
    output = []
    for pair in matched_pairs:
        pair_data = []
        for r_name in ["radar1", "radar2"]:
            radar_data = pair[f"{r_name}_data"]
            inliers = pair[f"{r_name}_inliers"]
            rev_idx = pair[f"{r_name}_revIdx"]

            # for k in ["rotation", "translation"]:
            #     radar_data[k] = torch.tensor(
            #         radar_data[k], dtype=torch.float32, device=device
            #     )
            for k in ["lineP0", "lineP1"]:
                # print("inliers[k]:", inliers[k])  # Debugging line
                # inliers[k] = torch.tensor(
                #     [inliers[k]["x"], inliers[k]["y"], inliers[k]["z"]],
                #     dtype=torch.float32,
                #     device=device,
                # )
                inliers[f"{k}_world"] = transform_points(
                    inliers[k].unsqueeze(0),
                    radar_data["rotation"],
                    radar_data["translation"],
                )[0]
            # inliers["inlierCloud"] = torch.tensor(
            #     [[p["x"], p["y"], p["z"]] for p in inliers["inlierCloud"]],
            #     dtype=torch.float32,
            #     device=device,
            # )
            inliers["inlierCloud_world"] = transform_points(
                inliers["inlierCloud"],
                radar_data["rotation"],
                radar_data["translation"],
            )

            pair_data.append(
                {"radar_data": radar_data, "inliers": inliers, "rev_idx": rev_idx}
            )
        output.append(pair_data)
    return output


def get_opt_transform(rev_idx, t, r, device):
    """
    Returns the translation and rotation matrix for a given rev_idx.
    If rev_idx > 0, uses the corresponding t and r; otherwise returns zeros/identity.
    """
    time0 = time.time()
    if rev_idx > 0:
        t_opt = t[rev_idx - 1]
        time1 = time.time()
        r_opt = r[rev_idx - 1]
        time2 = time.time()
        r_opt = axis_angle_to_matrix(r_opt)  # significant time
        # print(f"r_opt type device: {r_opt.dtype} {r_opt.device}") #r_opt type device: torch.float32 cuda:0

    else:
        t_opt = torch.zeros(3, device=device)
        time1 = time.time()
        r_opt = torch.eye(3, device=device)
        time2 = time.time()
    time3 = time.time()
    total_time = time3 - time0
    # print(
    #     f"{rev_idx} get_opt_transform total time: {total_time:.4f}s, first part: {time1 - time0:.4f}s, second part: {time2 - time1:.4f}s (creating rotation matrix: {time3 - time2:.4f}s"
    # )
    # 2 get_opt_transform total time: 0.0014s, first part: 0.0000s, second part: 0.0000s (creating rotation matrix: 0.0014s
    return t_opt, r_opt


def calculate_point_to_line_loss(r, t, preprocessed_matched_pairs):
    device = r.device
    loss = torch.tensor(0.0, device=device)
    for j, pair_data in tqdm(enumerate(preprocessed_matched_pairs), leave=False):
        # pair is a list of two dictionaries, one for each radar
        for i, data in enumerate(pair_data):
            time0 = time.time()
            other_idx = 1 - i

            t_opt, r_opt = get_opt_transform(data["rev_idx"], t, r, device)
            t_opt_other, r_opt_other = get_opt_transform(
                pair_data[other_idx]["rev_idx"], t, r, device
            )
            time1 = time.time()
            d = point_to_line_distance(
                transform_points(
                    pair_data[i]["inliers"]["inlierCloud_world"], r_opt, t_opt
                ),
                transform_points(
                    pair_data[other_idx]["inliers"]["lineP0_world"].unsqueeze(0),
                    r_opt_other,
                    t_opt_other,
                )[0],
                transform_points(
                    pair_data[other_idx]["inliers"]["lineP1_world"].unsqueeze(0),
                    r_opt_other,
                    t_opt_other,
                )[0],
            )
            time2 = time.time()
            d = d.mean()
            time3 = time.time()
            loss = loss + d
            time4 = time.time()
            total_time = time4 - time0
            print(
                f"total time {total_time:.4f}s, "
                f"timing breakdown: data extraction: {time1 - time0:.4f}s, "
                f"distance calculation: {time2 - time1:.4f}s, "
                f"mean calculation: {time3 - time2:.4f}s, "
                f"loss update: {time4 - time3:.4f}s"
            )
            # total time 0.0034s, timing breakdown: data extraction: 0.0028s, distance calculation: 0.0006s, mean calculation: 0.0000s, loss update: 0.0000s
        if False:
            # use torchviz
            params = {
                "r": r,
                "t": t,
                "d": d,
                "loss": loss,
                "r_opt": r_opt,
                "t_opt": t_opt,
                # "line_vector": line_vector,
                # "line_vector_normalized": line_vector_normalized,
            }
            dot = make_dot(loss, params=params, show_attrs=True, max_attr_chars=50)
            dot.render(
                f"/home/palakons/logs/r-r_matching/graph/graph_{j:06d}",
                format="png",
            )
    return loss / len(matched_pairs)


def load_inliers(inlier_fname, point_index):
    with open(inlier_fname, "r") as f:
        inliers_list = json.load(f)
    return (
        inliers_list["lines"][point_index]
        if point_index < len(inliers_list["lines"])
        else None
    )


def test_calculate_point_to_line_loss():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    line_p0 = torch.randn(3, device=device)
    line_p1 = line_p0 + torch.tensor([0.0, 1.0, 0.0], device=device)
    points = line_p0 + torch.linspace(-1, 2, 100, device=device)[:, None] * (
        line_p1 - line_p0
    )

    rotation = torch.eye(3, device=device)
    translation = torch.zeros(3, device=device)

    matched_pairs = [
        {
            "radar1_data": {"rotation": rotation, "translation": translation},
            "radar2_data": {"rotation": rotation, "translation": translation},
            "radar1_inliers": {
                "inlierCloud": points,
                "lineP0": line_p0,
                "lineP1": line_p1,
            },
            "radar2_inliers": {
                "inlierCloud": points,
                # shift x by 1
                "lineP0": line_p0 + torch.tensor([1.0, 0.0, 0.0], device=device),
                "lineP1": line_p1 + torch.tensor([1.0, 0.0, 0.0], device=device),
            },
            "radar1_revIdx": 0,
            "radar2_revIdx": 1,
        }
    ]
    r = torch.zeros(5, 3, device=device, requires_grad=True)
    t = torch.zeros(5, 3, device=device, requires_grad=True)

    loss = calculate_point_to_line_loss(r, t, matched_pairs)
    print(f"Test loss (should be zero): {loss.item()}")
    assert torch.isclose(
        loss, torch.tensor(0.0, device=device), atol=1e-6
    ), f"Loss should be zero for points on the line, {loss.item()}"


def reverse_index(radar_channel):
    """
    Reverse the index of the radar channel.
    This is a placeholder function, implement the actual logic as needed.
    """
    radar_chs = [
        "RADAR_LEFT_FRONT",
        "RADAR_LEFT_BACK",
        "RADAR_RIGHT_FRONT",
        "RADAR_RIGHT_BACK",
        "RADAR_LEFT_SIDE",
        "RADAR_RIGHT_SIDE",
    ]
    if radar_channel in radar_chs:
        return radar_chs.index(radar_channel)
    else:
        raise ValueError(f"Unknown radar channel: {radar_channel}")


def read_and_reformat_scene_data(scene_data_fname, device):
    with open(scene_data_fname, "r") as f:
        scene_data = json.load(f)

    # Reformatting the scene data to match the expected structure
    reformatted_data = {}
    # print(        "scene data keys:", scene_data[0].keys()    )  # keys: dict_keys(['seq_id', 'frame_id', 'RADAR_LEFT_FRONT', 'RADAR_LEFT_BACK', 'RADAR_RIGHT_FRONT', 'RADAR_RIGHT_BACK', 'RADAR_LEFT_SIDE', 'RADAR_RIGHT_SIDE', 'CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_FRONT', 'CAMERA_RIGHT_BACK'])
    for scene_item in scene_data:
        seq_id = scene_item["seq_id"]
        frame_id = scene_item["frame_id"]
        if seq_id not in reformatted_data:
            reformatted_data[seq_id] = {}
        if frame_id not in reformatted_data[seq_id]:
            reformatted_data[seq_id][frame_id] = {}

        for radar_key in [
            "RADAR_LEFT_FRONT",
            "RADAR_LEFT_BACK",
            "RADAR_RIGHT_FRONT",
            "RADAR_RIGHT_BACK",
            "RADAR_LEFT_SIDE",
            "RADAR_RIGHT_SIDE",
        ]:
            if radar_key in scene_item:
                reformatted_data[seq_id][frame_id][radar_key] = {
                    "rotation": scene_item[radar_key]["rotation"],
                    "translation": scene_item[radar_key]["translation"],
                    "image_file": scene_item[radar_key]["image_file"],
                }

    return reformatted_data


if False:  # testing
    test_calculate_point_to_line_loss()  # Run a test to verify the function works correctly
    exit()

# set seq_id and frame_id
parser = argparse.ArgumentParser(description="Process seq_id and frame_id.")
parser.add_argument("--step", type=int, default=1000, help="Optimization step size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

parser.add_argument(
    "--r_reg_weight",
    type=float,
    default=0.00,
    help="Regularization weight for rotation (r)",
)
parser.add_argument(
    "--t_reg_weight",
    type=float,
    default=0.00,
    help="Regularization weight for translation (t)",
)
args = parser.parse_args()

opt_step = args.step
opt_lr = args.lr
r_reg_weight = args.r_reg_weight
t_reg_weight = args.t_reg_weight


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_run_suffix(lr, r_reg_weight, t_reg_weight):
    return f"lr{lr:.4g}_rreg{r_reg_weight:.4g}_treg{t_reg_weight:.4g}"


run_suffix = get_run_suffix(opt_lr, r_reg_weight, t_reg_weight)
check_point_path = (
    f"/home/palakons/logs/r-r_matching/check_point_{run_suffix}_30_refactor.pt"
)

if os.path.exists(check_point_path):
    print(f"Loading checkpoint from {check_point_path}...")
    checkpoint = torch.load(check_point_path, map_location=device)
    start_step = checkpoint["step"]
    r = torch.tensor(
        checkpoint["r"], dtype=torch.float32, device=device, requires_grad=True
    )
    t = torch.tensor(
        checkpoint["t"], dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    loss_history = checkpoint.get("loss_history", [])
    main_loss_history = checkpoint.get("main_loss_history", [])
    reg_loss_history = checkpoint.get("reg_loss_history", [])
    r_hist = checkpoint.get("r_hist", [])
    t_hist = checkpoint.get("t_hist", [])
    r_reg_weight = checkpoint.get("r_reg_weight", r_reg_weight)
    t_reg_weight = checkpoint.get("t_reg_weight", t_reg_weight)
    print(
        f"Resuming from step {start_step}, r: {r}, t: {t}, lr: {opt_lr}, r_reg_weight: {r_reg_weight}, t_reg_weight: {t_reg_weight}"
    )
else:
    optimizer_state_dict = None
    loss_history = []
    main_loss_history = []
    reg_loss_history = []
    r_hist = []
    t_hist = []

    print("No checkpoint found, starting from scratch.", check_point_path)
    start_step = 0

    # # declare of random r t, the optimized paramteres for 5 radars
    # r = torch.randn(
    #     5, 3, device=device, requires_grad=True
    # )  # 5 radars, 3 parameters each (rotation)
    # t = torch.randn(
    #     5, 3, device=device, requires_grad=True
    # )  # 5 radars, 3 parameters each (translation)
    r = torch.zeros(5, 3, device=device, requires_grad=True)
    t = torch.zeros(5, 3, device=device, requires_grad=True)

    loss_history = []
    main_loss_history = []
    reg_loss_history = []
    r_hist = []
    t_hist = []

tt = trange(start_step, opt_step)

optimizer = torch.optim.Adam([r, t], lr=opt_lr)
optimizer.load_state_dict(optimizer_state_dict) if optimizer_state_dict else None
matched_pairs = []

# params from /home/palakons/linear_hough_man_radar/public/man_scene_data.json to ram

scene_data_fname = "/home/palakons/linear_hough_man_radar/public/man_scene_data.json"


print(f"Loading scene data from {scene_data_fname}...")
scene_data = read_and_reformat_scene_data(scene_data_fname, device=device)
# scene_data: {seq_id: {frame_id: {radar_ch: {rotation, translation, image_file}}}}

# matched pairs from /home/palakons/linear_hough_man_radar/public/matched_pairs.csv
print(
    "Loading matched pairs from /home/palakons/linear_hough_man_radar/public/matched_pairs_30.csv..."
)
matched_pairs = pd.read_csv(
    "/home/palakons/linear_hough_man_radar/public/matched_pairs_30.csv"
)
# matched_pairs: [{seq_id,frame_id,radar1,line1Index,line1Inliers,radar2,line2Index,line2Inliers}}]
# go through each mathced pairs
matched_pairs = matched_pairs.to_dict(orient="records")

# inlier from /ist-nas/users/palakonk/singularity/home/palakons/linear_hough_man_radar/public/json
print("processing matched pairs...")
for i in trange(len(matched_pairs)):
    # print("matched_pairs[i]:", matched_pairs[i]) #matched_pairs[i]: {'seq_id': 1, 'frame_id': 39, 'radar1': 'RADAR_RIGHT_FRONT', 'line1Index': 0, 'line1Inliers': 64, 'radar2': 'RADAR_LEFT_SIDE', 'line2Index': 0, 'line2Inliers': 66}
    for j in [1, 2]:
        inlier_fname = (
            scene_data[matched_pairs[i]["seq_id"]][matched_pairs[i]["frame_id"]][
                matched_pairs[i][f"radar{j}"]
            ]["image_file"]
            .split("/")[-1]
            .replace(".pcd", ".json")
        )
        # print(
        #     f"Loading inliers for {matched_pairs[i]['seq_id']} {matched_pairs[i]['frame_id']} radar{j} from {inlier_fname}"
        # )  # samples/RADAR_RIGHT_FRONT/RADAR_RIGHT_FRONT_1696700144004218.pcd
        matched_pairs[i][f"radar{j}_data"] = scene_data[matched_pairs[i]["seq_id"]][
            matched_pairs[i]["frame_id"]
        ][matched_pairs[i][f"radar{j}"]]
        matched_pairs[i][f"radar{j}_data"]["rotation"] = torch.tensor(
            matched_pairs[i][f"radar{j}_data"]["rotation"],
            dtype=torch.float32,
            device=device,
        )
        matched_pairs[i][f"radar{j}_data"]["translation"] = torch.tensor(
            matched_pairs[i][f"radar{j}_data"]["translation"],
            dtype=torch.float32,
            device=device,
        )

        inliers = load_inliers(
            f"{line_json_root}/{inlier_fname}", matched_pairs[i][f"line{j}Index"]
        )  # Load inliers for each radar channel
        # inliers: {lineP0, lineP1, inlierCloud}
        matched_pairs[i][f"radar{j}_inliers"] = inliers

        # make lineP0/1  and inlierCloud  as tensor on device ad float32 (no more "x", "y", "z" keys)
        if inliers is not None:
            matched_pairs[i][f"radar{j}_inliers"]["lineP0"] = torch.tensor(
                [
                    inliers["lineP0"]["x"],
                    inliers["lineP0"]["y"],
                    inliers["lineP0"]["z"],
                ],
                dtype=torch.float32,
                device=device,
            )
            matched_pairs[i][f"radar{j}_inliers"]["lineP1"] = torch.tensor(
                [
                    inliers["lineP1"]["x"],
                    inliers["lineP1"]["y"],
                    inliers["lineP1"]["z"],
                ],
                dtype=torch.float32,
                device=device,
            )
            matched_pairs[i][f"radar{j}_inliers"]["inlierCloud"] = torch.tensor(
                [[p["x"], p["y"], p["z"]] for p in inliers["inlierCloud"]],
                dtype=torch.float32,
                device=device,
            )

        matched_pairs[i][f"radar{j}_revIdx"] = reverse_index(
            matched_pairs[i][f"radar{j}"]
        )

# print("matched_pairs[i]:", matched_pairs[i])

# matched_pairs[i]: {
#     "seq_id": 3,
#     "frame_id": 39,
#     "radar1": "RADAR_LEFT_FRONT",
#     "line1Index": 0,
#     "line1Inliers": 93,
#     "radar2": "RADAR_RIGHT_BACK",
#     "line2Index": 0,
#     "line2Inliers": 64,
#     "radar1_data": {
#         "rotation": tensor(
#             [
#                 [0.9414, 0.3353, 0.0370],
#                 [-0.3353, 0.9421, -0.0055],
#                 [-0.0367, -0.0072, 0.9993],
#             ],
#             device="cuda:0",
#         ),
#         "translation": tensor([5.2590, 1.2266, 2.0126], device="cuda:0"),
#         "image_file": "samples/RADAR_LEFT_FRONT/RADAR_LEFT_FRONT_1692868191204085.pcd",
#     },
#     "radar1_inliers": {
#         "p0": {"x": 8.95732, "y": -15.0739, "z": -1.79428},
#         "p1": {"x": 160.011, "y": 36.0641, "z": 1.12562},
#         "inliers": 93,
#         "lapseTime": 0.82077,
#         "lineP0": tensor([69.1218, 5.2866, -0.6331], device="cuda:0"),
#         "lineP1": tensor([70.0692, 5.6060, -0.6139], device="cuda:0"),
#         "inlierCloud": tensor(
#             [
#                 [8.9573e00, -1.5074e01, -1.7943e00],
#                 ...
#                 [1.7705e02, 4.1909e01, 1.9115e00],
#             ],
#             device="cuda:0",
#         ),
#     },
#     "radar1_revIdx": 0,
#     "radar2_data": {
#         "rotation": tensor(
#             [
#                 [-0.7663, -0.6423, 0.0099],
#                 [-0.6423, 0.7664, 0.0105],
#                 [-0.0144, 0.0016, -0.9999],
#             ],
#             device="cuda:0",
#         ),
#         "translation": tensor([5.0062, -1.3964, 2.0017], device="cuda:0"),
#         "image_file": "samples/RADAR_RIGHT_BACK/RADAR_RIGHT_BACK_1692868191202970.pcd",
#     },
#     "radar2_inliers": {
#         "p0": {"x": 11.6582, "y": -9.16355, "z": 1.71541},
#         "p1": {"x": 74.6767, "y": 43.5326, "z": 0.427887},
#         "inliers": 64,
#         "lapseTime": 0.229026,
#         "lineP0": tensor([39.2793, 14.0484, 1.1439], device="cuda:0"),
#         "lineP1": tensor([40.0485, 14.6872, 1.1281], device="cuda:0"),
#         "inlierCloud": tensor(
#             [
#                 [9.7968e00, -1.0646e01, 1.7199e00],
#                 ...
#                 [8.6840e01, 5.3802e01, 2.6212e-01],
#             ],
#             device="cuda:0",
#         ),
#     },
#     "radar2_revIdx": 3,
# }

# Preprocess matched pairs for optimization
preprocessed_matched_pairs = preprocess_matched_pairs(matched_pairs, device=device)
time7 = time.time()
for i_step in tt:
    time0 = time.time()
    print("time between loop:", time7 - time0)
    optimizer.zero_grad()
    time1 = time.time()
    # main_loss = calculate_point_to_line_loss_tower(r, t, matched_pairs)
    main_loss = calculate_point_to_line_loss(r, t, preprocessed_matched_pairs)
    time2 = time.time()
    reg_loss = r_reg_weight * r.norm() + t_reg_weight * t.norm()
    loss = main_loss + reg_loss
    time3 = time.time()
    loss_history.append(loss.item())
    main_loss_history.append(main_loss.item())
    reg_loss_history.append(reg_loss.item())
    time4 = time.time()
    r_hist.append(r.detach().cpu().numpy().tolist())
    t_hist.append(t.detach().cpu().numpy().tolist())
    time5 = time.time()
    loss.backward()
    tt.set_description(f"Loss: {loss.item():.4f}")
    time6 = time.time()
    optimizer.step()
    time7 = time.time()

    # print(
    #     f"[Timing] Step {i_step}: Zero grad: {time1-time0:.4f}s, Main loss: {time2-time1:.4f}s, Reg loss: {time3-time2:.4f}s, Backward: {time7-time6:.4f}s, Step: {time7-time0:.4f}s appending r and t: {time6-time5:.4f}s appending loss: {time4-time3:.4f}s"
    # )  # [
    # print time each part in percent of total time
    total_time = time7 - time0
    print(
        f"Total time for step {i_step}: {total_time:.4f}s   "
        f"Zero grad: {(time1-time0)/total_time:.2%}, "
        f"Main loss: {(time2-time1)/total_time:.2%}, "
        f"Reg loss: {(time3-time2)/total_time:.2%}, "
        f"Appending loss: {(time4-time3)/total_time:.2%}, "
        f"Appending r and t: {(time5-time4)/total_time:.2%}, "
        f"Backward: {(time6-time5)/total_time:.2%}, "
        f"Optimizer step: {(time7-time6)/total_time:.2%}"
    )  # Total time for step 0: 0.1234s   Zero grad: 10.00%, Main loss: 20.00%, Reg loss: 30.00%, Backward: 40.00%, Appending r and t: 50.00%, Appending loss: 60.00%
    if i_step % 1000 == 0 or i_step == opt_step - 1:
        print("r:", r)
        print("t:", t)
        # save loss curve
        loss_curve_file_path = (
            f"/home/palakons/logs/r-r_matching/loss_curve_{run_suffix}_30_refactor.png"
        )

        fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

        # Loss curves
        axs[0].plot(
            range(len(main_loss_history)),
            main_loss_history,
            label="Main Loss",
            color="tab:blue",
            linestyle="-",
            linewidth=2,
        )
        axs[0].plot(
            range(len(reg_loss_history)),
            reg_loss_history,
            label="Reg Loss",
            color="tab:orange",
            linestyle="--",
            linewidth=2,
        )
        axs[0].plot(
            range(len(loss_history)),
            loss_history,
            label="Total Loss",
            color="tab:green",
            linestyle="-.",
            linewidth=2,
        )
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Loss")
        axs[0].set_yscale("log")
        axs[0].grid(True)
        axs[0].set_title(
            f"Loss Curves, LR: {opt_lr}, r_reg: {r_reg_weight}, t_reg: {t_reg_weight}, Step: {i_step} 50 inlier minimum"
        )
        axs[0].annotate(
            f"Step {i_step}\nTotal: {loss.item():.4f}\nMain: {main_loss.item():.4f}\nReg: {reg_loss.item():.4f}",
            xy=(len(loss_history) - 1, loss_history[-1]),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=10,
        )
        axs[0].legend()

        # r_hist: shape (steps, 5, 3)
        radar_chs = [
            "RADAR_LEFT_BACK",
            "RADAR_RIGHT_FRONT",
            "RADAR_RIGHT_BACK",
            "RADAR_LEFT_SIDE",
            "RADAR_RIGHT_SIDE",
        ]
        r_hist_np = np.array(r_hist)
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
        ]
        linestyles = ["-", "--", "-."]
        linewidths = [2, 2, 2]
        for radar_idx, radar_ch in enumerate(radar_chs):
            for comp_idx, comp_name in enumerate(["x", "y", "z"]):
                axs[1].plot(
                    r_hist_np[:, radar_idx, comp_idx],
                    label=f"{radar_ch}_{comp_name}",
                    color=colors[radar_idx],
                    linestyle=linestyles[comp_idx],
                    linewidth=linewidths[comp_idx],
                )
        axs[1].set_ylabel("Angle (rad)")
        axs[1].set_title("Rotation (axis-angle) history")
        axs[1].legend(ncol=3, fontsize=8)
        axs[1].grid(True)

        # t_hist: shape (steps, 5, 3)
        t_hist_np = np.array(t_hist)
        for radar_idx, radar_ch in enumerate(radar_chs):
            for comp_idx, comp_name in enumerate(["x", "y", "z"]):
                axs[2].plot(
                    t_hist_np[:, radar_idx, comp_idx],
                    label=f"{radar_ch}_{comp_name}",
                    color=colors[radar_idx],
                    linestyle=linestyles[comp_idx],
                    linewidth=linewidths[comp_idx],
                )
        axs[2].set_xlabel("Step")
        axs[2].set_ylabel("Translation (m)")
        axs[2].set_title("Translation history")
        axs[2].legend(ncol=3, fontsize=8)
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(loss_curve_file_path)
        plt.close()

        torch.save(
            {
                "step": i_step,
                "r": r.detach().cpu().numpy().tolist(),
                "t": t.detach().cpu().numpy().tolist(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "main_loss_history": main_loss_history,
                "reg_loss_history": reg_loss_history,
                "r_hist": r_hist,
                "t_hist": t_hist,
                "r_reg_weight": r_reg_weight,
                "t_reg_weight": t_reg_weight,
            },
            check_point_path,
        )

        # output all r and t at all time steps for visualization
        # r_hist and t_hist: list of [step][radar][component]
        r_t_json = {
            "steps": [],
            "lr": opt_lr,
            "r_reg_weight": r_reg_weight,
            "t_reg_weight": t_reg_weight,
        }
        radar_chs = [
            "RADAR_LEFT_FRONT",
            "RADAR_LEFT_BACK",
            "RADAR_RIGHT_FRONT",
            "RADAR_RIGHT_BACK",
            "RADAR_LEFT_SIDE",
            "RADAR_RIGHT_SIDE",
        ]
        for step_idx in range(0, len(r_hist), 10):
            step_entry = {
                "step": step_idx,
                "overall_loss": loss_history[step_idx],
                "main_loss": main_loss_history[step_idx],
                "reg_loss": reg_loss_history[step_idx],
            }
            for radar_idx, radar_ch in enumerate(radar_chs):
                if radar_idx == 0:
                    step_entry[radar_ch] = {
                        "rotation": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "translation": [0.0, 0.0, 0.0],
                    }
                else:
                    axis_angle = torch.tensor(r_hist[step_idx][radar_idx - 1])
                    rot_matrix = (
                        axis_angle_to_matrix(axis_angle).detach().cpu().numpy().tolist()
                    )
                    step_entry[radar_ch] = {
                        "rotation": rot_matrix,
                        "translation": t_hist[step_idx][radar_idx - 1],
                    }
            r_t_json["steps"].append(step_entry)
        with open(
            f"/home/palakons/logs/r-r_matching/r_t_optimized_{run_suffix}_30_refactor.json",
            "w",
        ) as f:
            json.dump(r_t_json, f, indent=2)
