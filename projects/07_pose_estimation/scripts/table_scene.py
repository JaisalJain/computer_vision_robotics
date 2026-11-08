"""A small synthetic robot work cell for the coordinate-frame tutorial (notebook 03).

Frames (all right-handed, metres, ROS REP-103 conventions except the optical frame):

* ``world``      - table centre on the table surface; x forward (away from the robot), y left, z up.
* ``tag<i>``     - four "map" AprilTags lying on the table (marker frame: z up, out of the tag).
* ``object``     - a box on the table; origin at the centre of its bottom face, z up.
* ``obj_tag``    - an AprilTag stuck on the box's top face (``T_object_objtag``).
* ``base_link``  - robot base, behind the table, x pointing at the table.
* ``camera_link``    - robot head camera body frame (REP-103: x forward, y left, z up).
* ``camera_optical`` - the same camera in OpenCV axes (x right, y down, z forward).

``T_a_b`` maps points from frame ``b`` into frame ``a``: ``p_a = T_a_b @ p_b``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import fiducial_sim as fs
import numpy as np

from cvlab.geometry import R_ROS_FROM_OPENCV, invert_T, look_at, make_T

TABLE_WIDTH = 1.2  # m (the table texture is a 4:3 photo -> 1.2 x 0.9 m)
MAP_TAG_SIZE = 0.10
MAP_TAGS = {0: (0.45, 0.30, 0.0), 1: (0.45, -0.30, 90.0), 2: (-0.45, -0.30, 180.0),
            3: (-0.45, 0.30, 270.0)}  # id -> (x, y, yaw_deg) in the world frame  # fmt: skip
OBJ_TAG_ID, OBJ_TAG_SIZE = 10, 0.06
BOX_SIZE = np.array([0.16, 0.10, 0.07])  # x, y, z extent of the box [m]


def Rx(deg: float) -> np.ndarray:  # noqa: N802
    return fs.rot_axis_angle([1.0, 0.0, 0.0], np.radians(deg))


def Ry(deg: float) -> np.ndarray:  # noqa: N802
    return fs.rot_axis_angle([0.0, 1.0, 0.0], np.radians(deg))


def Rz(deg: float) -> np.ndarray:  # noqa: N802
    return fs.rot_axis_angle([0.0, 0.0, 1.0], np.radians(deg))


@dataclass
class Scene:
    T_world_tag: dict[int, np.ndarray]
    T_world_object: np.ndarray
    T_object_objtag: np.ndarray
    T_world_base: np.ndarray
    T_base_camlink: np.ndarray
    T_camlink_optical: np.ndarray
    items: list = field(default_factory=list)  # (texture, T_world_plane) in paint order
    tag_size: dict[int, float] = field(default_factory=dict)

    @property
    def T_world_optical(self) -> np.ndarray:  # noqa: N802
        """Head camera pose in the world (camera_optical -> world)."""
        return self.T_world_base @ self.T_base_camlink @ self.T_camlink_optical

    @property
    def T_world_objtag(self) -> np.ndarray:  # noqa: N802
        return self.T_world_object @ self.T_object_objtag


def make_scene(table_photo: np.ndarray, object_xy_yaw=(0.05, -0.05, 30.0)) -> Scene:
    """Build the work cell. ``table_photo`` (HxWx3 uint8) is used as the table-top texture."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    items = [(fs.image_texture(table_photo, TABLE_WIDTH), np.eye(4))]
    T_world_tag, tag_size = {}, {}
    for i, (x, y, yaw) in MAP_TAGS.items():
        T_world_tag[i] = make_T(Rz(yaw), [x, y, 0.0])
        tag_size[i] = MAP_TAG_SIZE
        items.append((fs.marker_texture(dictionary, i, MAP_TAG_SIZE), T_world_tag[i]))
    x, y, yaw = object_xy_yaw
    T_world_object = make_T(Rz(yaw), [x, y, 0.0])
    T_object_objtag = make_T(np.eye(3), [0.0, 0.0, BOX_SIZE[2]])  # tag on the top face
    tag_size[OBJ_TAG_ID] = OBJ_TAG_SIZE
    # The box's visible top face (light grey) is painted under the tag.
    top = np.full((100, 160), 200, np.uint8)
    top_tex = fs.image_texture(top, BOX_SIZE[0])
    items.append((top_tex, T_world_object @ T_object_objtag))
    items.append((fs.marker_texture(dictionary, OBJ_TAG_ID, OBJ_TAG_SIZE),
                  T_world_object @ T_object_objtag))  # fmt: skip
    return Scene(
        T_world_tag=T_world_tag,
        T_world_object=T_world_object,
        T_object_objtag=T_object_objtag,
        T_world_base=make_T(np.eye(3), [-0.80, 0.0, 0.0]),
        # Head camera 0.75 m above the base, pitched down by 40 deg (a +40 deg rotation about y
        # tilts the x (viewing) axis downwards).
        T_base_camlink=make_T(Ry(40.0), [0.05, 0.0, 0.75]),
        # REP-103 camera_link <- OpenCV optical frame: p_link = R_ROS_FROM_OPENCV @ p_optical.
        T_camlink_optical=make_T(R_ROS_FROM_OPENCV, np.zeros(3)),
        items=items,
        tag_size=tag_size,
    )


def orbit_trajectory(n: int = 60, radius: float = 1.05, height: float = 0.75,
                     az_range=(-60.0, 60.0)) -> np.ndarray:  # fmt: skip
    """``n`` camera poses ``T_world_cam`` (OpenCV axes) on an arc around the table.

    Azimuth 0 is behind the robot side of the table (-x); the camera looks at a point that
    wobbles slightly around the table centre so the motion is not perfectly regular.
    """
    poses = []
    for k, az in enumerate(np.linspace(*az_range, n)):
        a = np.radians(180.0 + az)
        eye = np.array([radius * np.cos(a), radius * np.sin(a), height + 0.05 * np.sin(k / 7)])
        target = np.array([0.05 * np.sin(k / 5), 0.05 * np.cos(k / 9), 0.0])
        poses.append(look_at(eye, target, up=(0.0, 0.0, 1.0)))
    return np.stack(poses)


def box_corners(size: np.ndarray = BOX_SIZE) -> np.ndarray:
    """8x3 corners of the box in the object frame (origin at the bottom-face centre)."""
    x, y, z = size / 2.0
    return np.array([[sx * x, sy * y, h] for h in (0.0, size[2]) for sx, sy in
                     ((-1, -1), (1, -1), (1, 1), (-1, 1))])  # fmt: skip


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]  # fmt: skip


def draw_box(img: np.ndarray, K: np.ndarray, T_cam_object: np.ndarray, color=(0, 220, 255),
             thickness: int = 2) -> np.ndarray:  # fmt: skip
    """Project and draw the box wireframe (in place) given its pose in the camera frame."""
    pts = box_corners() @ T_cam_object[:3, :3].T + T_cam_object[:3, 3]
    uv = (pts @ K.T)[:, :2] / pts[:, 2:3]
    for i, j in BOX_EDGES:
        cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int)),
                 color, thickness, cv2.LINE_AA)  # fmt: skip
    return img


def map_points(scene: Scene, ids) -> np.ndarray:
    """World coordinates of the 4 corners of each map tag in ``ids`` (IPPE_SQUARE order)."""
    out = []
    for i in ids:
        T = scene.T_world_tag[i]
        out.append(fs.marker_object_points(scene.tag_size[i]) @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(out)


def localize_joint(scene: Scene, detections: dict[int, np.ndarray], K: np.ndarray):
    """Camera pose ``T_world_cam`` from *all* visible map tags with one PnP (SQPnP + LM).

    Returns ``None`` when no map tag is visible.
    """
    ids = [i for i in sorted(detections) if i in scene.T_world_tag]
    if not ids:
        return None
    obj = map_points(scene, ids)
    img = np.concatenate([detections[i] for i in ids]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    T_cam_world = make_T(cv2.Rodrigues(rvec)[0], tvec.ravel())
    return invert_T(T_cam_world)


def localize_single(scene: Scene, detections: dict[int, np.ndarray], K: np.ndarray):
    """Camera pose from the single largest visible map tag (IPPE_SQUARE). ``None`` if none."""
    ids = [i for i in detections if i in scene.T_world_tag]
    if not ids:
        return None
    area = {i: cv2.contourArea(detections[i].astype(np.float32)) for i in ids}
    best = max(ids, key=area.get)
    res = fs.solve_marker_pose(detections[best], scene.tag_size[best], K)
    T_cam_tag = make_T(res.R, res.t)
    return scene.T_world_tag[best] @ invert_T(T_cam_tag)
