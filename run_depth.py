# run depth
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated, Callable

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from tqdm import tqdm

from flow3d.data import DavisDataConfig, get_train_val_datasets, iPhoneDataConfig, CustomDataConfig
from flow3d.renderer import Renderer
from flow3d.trajectories import (
    get_arc_w2cs,
    get_avg_w2c,
    get_lemniscate_w2cs,
    get_lookat,
    get_spiral_w2cs,
    get_wander_w2cs,
)
from flow3d.vis.utils import make_video_divisble

from run_video import VideoConfig, TrainTrajectoryConfig, OptTrainTrajectoryConfig

torch.set_float32_matmul_precision("high")



def main(cfg: VideoConfig):
    train_dataset = get_train_val_datasets(cfg.data, load_val=False)[0]
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path)

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir=cfg.work_dir,
        port=None,
    )
    assert train_dataset.num_frames == renderer.num_frames

    guru.info(f"Rendering video from {renderer.global_step=}")

    train_w2cs = train_dataset.get_w2cs().to(device)
    avg_w2c = get_avg_w2c(train_w2cs)
    # avg_w2c = train_w2cs[0]
    train_c2ws = torch.linalg.inv(train_w2cs)
    lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    K = train_dataset.get_Ks()[0].to(device)
    img_wh = train_dataset.get_img_wh()

    # get the radius of the bounding sphere of training cameras
    rc_train_c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(avg_w2c), train_c2ws)
    rc_pos = rc_train_c2ws[:, :3, -1]
    rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 1.25

    if isinstance(cfg.trajectory, TrainTrajectoryConfig):
        w2cs = cfg.trajectory.get_w2cs(train_w2cs=train_w2cs)
    elif isinstance(cfg.trajectory, OptTrainTrajectoryConfig):
        w2cs = cfg.trajectory.get_w2cs(train_w2cs=renderer.model.camera.get_w2cs().detach())        
    else:
        w2cs = cfg.trajectory.get_w2cs(
            ref_w2c=(
                avg_w2c
                if cfg.trajectory.ref_t < 0
                else train_w2cs[min(cfg.trajectory.ref_t, train_dataset.num_frames - 1)]
            ),
            lookat=lookat,
            up=up,
            focal_length=K[0, 0].item(),
            rads=rads,
        )
    ts = cfg.time.get_ts(
        num_frames=renderer.num_frames,
        traj_frames=cfg.trajectory.num_frames,
        device=device,
    )

    import viser.transforms as vt
    from flow3d.vis.utils import get_server

    server = get_server(port=8890)
    for i, train_w2c in enumerate(train_w2cs):
        train_c2w = torch.linalg.inv(train_w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/train_camera/{i:03d}",
            np.pi / 4,
            1.0,
            0.02,
            color=(0, 0, 0),
            wxyz=vt.SO3.from_matrix(train_c2w[:3, :3]).wxyz,
            position=train_c2w[:3, -1],
        )
    for i, w2c in enumerate(w2cs):
        c2w = torch.linalg.inv(w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/camera/{i:03d}",
            np.pi / 4,
            1.0,
            0.02,
            color=(255, 0, 0),
            wxyz=vt.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, -1],
        )
        avg_c2w = torch.linalg.inv(avg_w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/ref_camera",
            np.pi / 4,
            1.0,
            0.02,
            color=(0, 0, 255),
            wxyz=vt.SO3.from_matrix(avg_c2w[:3, :3]).wxyz,
            position=avg_c2w[:3, -1],
        )
    import ipdb

    ipdb.set_trace()

    # num_frames = len(train_w2cs)
    # w2cs = train_w2cs[:1].repeat(num_frames, 1, 1)
    # ts = torch.arange(num_frames, device=device)
    # assert len(w2cs) == len(ts)

    video = []
    for w2c, t in zip(tqdm(w2cs), ts):
        with torch.inference_mode():
            img = renderer.model.render(int(t.item()), w2c[None], K[None], img_wh,
                                        return_depth=True)[
                "depth"
            ][0]
        img = 1 / img
        img = torch.clamp(img, 0, 1)
        img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        video.append(img)
    video = np.stack(video, 0)

    video_dir = f"{cfg.work_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(video_dir, exist_ok=True)
    iio.imwrite(f"{video_dir}/depth.mp4", make_video_divisble(video), fps=cfg.fps)
    with open(f"{video_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)


if __name__ == "__main__":
    main(tyro.cli(VideoConfig))
