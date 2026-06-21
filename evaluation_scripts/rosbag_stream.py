"""
ROS bag → DROID-SLAM color / RGB-D stream.

Uses rosbags (already installed via evo: pip install rosbags), so no ROS
installation or sourcing is needed. Works with Python 3.9 in the venv.

Frames are read lazily from the bag — no in-memory caching — so sequences
with 20 000+ frames are handled without excessive RAM usage. The stream can
be iterated multiple times (e.g. once for track(), once for terminate()) and
re-reads the bag each time, matching demo.py's generator pattern.

Both stream classes accept either a single bag path (str) or a list of paths
that are iterated sequentially — use a list when one logical sequence is split
across multiple bag files for size control.
"""

from collections import deque

import cv2
import numpy as np
import torch

from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores

_store = get_typestore(Stores.ROS1_NOETIC)


# ── Color decode helpers ───────────────────────────────────────────────────────

def _decode_compressed(msg):
    """sensor_msgs/CompressedImage → BGR uint8 numpy array."""
    return cv2.imdecode(msg.data, cv2.IMREAD_COLOR)


def _decode_raw(msg):
    """sensor_msgs/Image → BGR uint8 numpy array."""
    img = msg.data.reshape(msg.height, msg.width, -1)
    if msg.encoding in ("rgb8", "RGB8"):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding in ("mono8",):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # bgr8 / BGR8: already correct
    return img


# ── Depth decode helpers ───────────────────────────────────────────────────────

def _decode_compressed_depth(msg):
    """
    sensor_msgs/CompressedImage (compressedDepth) → float32 metres.

    ROS image_transport/compressedDepth prepends a 12-byte header before the
    PNG payload: 4 bytes config-type + 4 bytes depth-quantisation float +
    4 bytes max-depth float. Skip all 12 before passing to imdecode.
    """
    data = np.frombuffer(msg.data.tobytes(), dtype=np.uint8)
    img = cv2.imdecode(data[12:], cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 1000.0   # mm → metres
    return img.astype(np.float32)


def _decode_raw_depth(msg):
    """sensor_msgs/Image (16UC1 or 32FC1 depth) → float32 metres."""
    if msg.encoding == "16UC1":
        return (msg.data.view(np.uint16)
                .reshape(msg.height, msg.width)
                .astype(np.float32) / 1000.0)  # mm → metres
    if msg.encoding == "32FC1":
        return msg.data.view(np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


# ── Depth alignment (for cameras with different intrinsics) ───────────────────

def _align_depth_to_color(depth_m, K_depth, K_color, h_color, w_color):
    """
    Remap depth from depth-camera frame into color-camera frame.

    Assumes the two cameras are co-located (same optical center) but have
    different focal lengths — the typical Gazebo simulation setup.  For each
    pixel (u_c, v_c) in the color image we compute the corresponding pixel
    in the depth image and sample it with nearest-neighbour interpolation.

    Back-project color pixel to 3-D (at unit depth), then project into
    depth camera:
        u_d = (u_c - cx_c) / fx_c * fx_d + cx_d
        v_d = (v_c - cy_c) / fy_c * fy_d + cy_d
    """
    fx_d, fy_d, cx_d, cy_d = K_depth
    fx_c, fy_c, cx_c, cy_c = K_color

    u_c = np.arange(w_color, dtype=np.float32)
    v_c = np.arange(h_color, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_c, v_c)

    map_x = (u_grid - cx_c) / fx_c * fx_d + cx_d
    map_y = (v_grid - cy_c) / fy_c * fy_d + cy_d

    return cv2.remap(depth_m, map_x, map_y,
                     cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0.0)


def _postprocess_depth(depth_m, h_out, w_out, K_depth=None, K_color=None, align=False):
    """
    Optionally align depth to color frame, then resize to (h_out, w_out).
    Zeros out NaN/inf and negative values.
    """
    if align and K_depth is not None and K_color is not None:
        h_c, w_c = depth_m.shape
        depth_m = _align_depth_to_color(depth_m, K_depth, K_color, h_c, w_c)

    if depth_m.shape[0] != h_out or depth_m.shape[1] != w_out:
        depth_m = cv2.resize(depth_m, (w_out, h_out), interpolation=cv2.INTER_NEAREST)

    depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0.0), depth_m, 0.0)
    return depth_m.astype(np.float32)


# ── Preprocessing (mirrors demo.py image_stream) ──────────────────────────────

_TARGET_AREA = (240, 320)  # default: matches demo.py --image_size; use (384,512) for higher quality


def _preprocess(bgr, intr, target_area=_TARGET_AREA):
    """
    Resize to ~target_area (preserving aspect ratio), crop to multiple of 8.
    Scale intrinsics [fx, fy, cx, cy] to match the resized image.

    Returns:
        image      : [1, 3, H, W] uint8 torch tensor (BGR channel order)
        intrinsics : [4] float32 torch tensor  [fx, fy, cx, cy]
        (h0,w0,h1,w1): original and final sizes (for depth resizing)
    """
    h0, w0 = bgr.shape[:2]
    scale = np.sqrt((target_area[0] * target_area[1]) / (h0 * w0))
    h1, w1 = int(h0 * scale), int(w0 * scale)
    bgr = cv2.resize(bgr, (w1, h1))
    bgr = bgr[: h1 - h1 % 8, : w1 - w1 % 8]
    h1, w1 = bgr.shape[:2]

    fx, fy, cx, cy = intr
    intrinsics = torch.tensor(
        [fx * w1 / w0, fy * h1 / h0, cx * w1 / w0, cy * h1 / h0],
        dtype=torch.float32,
    )
    image = torch.as_tensor(bgr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return image, intrinsics, (h0, w0, h1, w1)


# ── Camera info helper ─────────────────────────────────────────────────────────

def read_camera_info(bagfile, topic="/camera/color/camera_info"):
    """Return [fx, fy, cx, cy] from the first CameraInfo message in the bag."""
    # Accept a list; read from the first bag only
    if isinstance(bagfile, (list, tuple)):
        bagfile = bagfile[0]
    with Reader(bagfile) as bag:
        conns = [c for c in bag.connections if c.topic == topic]
        if not conns:
            return None
        for _, _, rawdata in bag.messages(connections=conns):
            msg = _store.deserialize_ros1(rawdata, conns[0].msgtype)
            # rosbags 0.9: CameraInfo uses uppercase K (3x3 matrix, row-major)
            return [float(msg.K[0]), float(msg.K[4]), float(msg.K[2]), float(msg.K[5])]
    return None


# ── Color-only stream ─────────────────────────────────────────────────────────

class RosbagColorStream:
    """
    Lazy color frame reader from one or more ROS bags in DROID-SLAM feed format.

    Accepts bagfile as a single path string or a list of paths — bags in the
    list are iterated sequentially (use a list when one sequence is split
    across multiple files for size control).

    Each call to __iter__ opens each bag in turn and reads frames on demand —
    no in-memory caching. Suitable for the second pass in droid.terminate().

    Handles:
      - Compressed topics (sensor_msgs/CompressedImage) — TSRB bags
      - Raw topics (sensor_msgs/Image)                  — Gazebo bags

    Yields: (timestamp, image[1,3,H,W] uint8, intrinsics[4] float32)
    """

    def __init__(
        self,
        bagfile,                           # str or list[str]
        color_topic: str,
        intrinsics,
        target_area: tuple = _TARGET_AREA,
        stride: int = 1,
        t0: float = 0.0,
    ):
        self.bagfiles = [bagfile] if isinstance(bagfile, str) else list(bagfile)
        self.color_topic = color_topic
        self.intrinsics = list(intrinsics)
        self.target_area = target_area
        self.stride = stride
        self.t0 = t0

        self._msgtype = None
        self._is_compressed = None
        self._approx_len = None
        self._image_size = None

    def _open_connections(self, bag, bagfile):
        conns = [c for c in bag.connections if c.topic == self.color_topic]
        if not conns:
            available = sorted(set(c.topic for c in bag.connections))
            raise RuntimeError(
                f"Topic '{self.color_topic}' not found in {bagfile}.\n"
                f"Available: {available}"
            )
        if self._msgtype is None:
            self._msgtype = conns[0].msgtype
            self._is_compressed = "Compressed" in self._msgtype
        return conns

    def _iter_frames(self):
        step = 0  # shared across all bags so stride is consistent end-to-end
        for bagfile in self.bagfiles:
            with Reader(bagfile) as bag:
                conns = self._open_connections(bag, bagfile)
                for _, ts_ns, rawdata in bag.messages(connections=conns):
                    tstamp = ts_ns * 1e-9
                    if tstamp < self.t0:
                        continue
                    if step % self.stride != 0:
                        step += 1
                        continue
                    try:
                        msg = _store.deserialize_ros1(rawdata, self._msgtype)
                        bgr = _decode_compressed(msg) if self._is_compressed else _decode_raw(msg)
                    except Exception as e:
                        print(f"[RosbagColorStream] decode failed at t={tstamp:.3f}: {e}")
                        step += 1
                        continue
                    if bgr is None:
                        step += 1
                        continue
                    image, intr, _ = _preprocess(bgr, self.intrinsics, self.target_area)
                    yield (tstamp, image, intr)
                    step += 1

    def __iter__(self):
        return self._iter_frames()

    def __len__(self):
        if self._approx_len is None:
            total = 0
            for bagfile in self.bagfiles:
                with Reader(bagfile) as bag:
                    conns = self._open_connections(bag, bagfile)
                    total += sum(c.msgcount for c in conns)
            self._approx_len = (total + self.stride - 1) // self.stride
        return self._approx_len

    def image_size(self):
        if self._image_size is None:
            frame = next(self._iter_frames())
            self._image_size = [frame[1].shape[2], frame[1].shape[3]]
        return self._image_size


# ── RGB-D stream ───────────────────────────────────────────────────────────────

class RosbagRGBDStream:
    """
    Lazy RGB-D frame reader that synchronises color and depth from one or more
    ROS bags played sequentially.

    Accepts bagfile as a single path string or a list of paths — bags are
    iterated in order (use a list when one sequence is split across multiple
    files for size control).

    Color and depth messages are read in a single chronological pass through
    each bag.  A small ring buffer holds recent depth frames so we can match
    them to color frames by nearest timestamp.  The pending-queue design
    handles two common orderings:
      TSRB   : depth arrives ~1 ms BEFORE the matching color frame
      Gazebo : color arrives ~4 ms BEFORE the matching depth frame

    Depth buffers are reset between bags (bags are always split on a complete
    frame boundary, so there is no cross-bag color/depth pairing needed).

    Depth alignment (Gazebo): depth and color cameras have different focal
    lengths but are co-located.  Each depth pixel is projected into the color
    camera frame using the ratio of focal lengths before resizing.

    Yields: (timestamp, image[1,3,H,W] uint8, depth[H,W] float32 metres, intrinsics[4])

    Pass a RosbagColorStream to droid.terminate() — terminate() only needs the
    color images and does not use depth.
    """

    # Maximum allowed time gap (seconds) between matched color/depth frames.
    MAX_SYNC_GAP_S = 0.05

    def __init__(
        self,
        bagfile,                       # str or list[str]
        color_topic: str,
        depth_topic: str,
        color_K,
        depth_K=None,
        align_depth: bool = False,
        target_area: tuple = _TARGET_AREA,
        stride: int = 1,
        t0: float = 0.0,
    ):
        self.bagfiles = [bagfile] if isinstance(bagfile, str) else list(bagfile)
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.color_K = list(color_K)
        self.depth_K = list(depth_K) if depth_K is not None else None
        self.align_depth = align_depth
        self.target_area = target_area
        self.stride = stride
        self.t0 = t0

        self._color_msgtype = None
        self._color_is_compressed = None
        self._depth_msgtype = None
        self._depth_is_compressed_depth = None
        self._approx_len = None
        self._image_size = None

    def _init_msgtypes(self, bag, bagfile):
        color_conns = [c for c in bag.connections if c.topic == self.color_topic]
        depth_conns = [c for c in bag.connections if c.topic == self.depth_topic]

        if not color_conns:
            raise RuntimeError(f"Color topic '{self.color_topic}' not found in {bagfile}")
        if not depth_conns:
            raise RuntimeError(f"Depth topic '{self.depth_topic}' not found in {bagfile}")

        if self._color_msgtype is None:
            self._color_msgtype = color_conns[0].msgtype
            self._color_is_compressed = "Compressed" in self._color_msgtype
        if self._depth_msgtype is None:
            self._depth_msgtype = depth_conns[0].msgtype
            self._depth_is_compressed_depth = "CompressedImage" in self._depth_msgtype

        return color_conns, depth_conns

    def _decode_depth_raw(self, rawdata):
        msg = _store.deserialize_ros1(rawdata, self._depth_msgtype)
        if self._depth_is_compressed_depth:
            return _decode_compressed_depth(msg)
        return _decode_raw_depth(msg)

    def _build_frame(self, c_ts_ns, c_raw, depth_m):
        """Decode buffered color rawdata + depth array into a yield tuple."""
        try:
            msg = _store.deserialize_ros1(c_raw, self._color_msgtype)
            bgr = _decode_compressed(msg) if self._color_is_compressed else _decode_raw(msg)
        except Exception as e:
            print(f"[RosbagRGBDStream] color decode error at t={c_ts_ns*1e-9:.3f}: {e}")
            return None
        if bgr is None:
            return None

        image, intr, (h0, w0, h1, w1) = _preprocess(bgr, self.color_K, self.target_area)

        depth_tensor = None
        if depth_m is not None:
            d_proc = _postprocess_depth(
                depth_m, h1, w1,
                K_depth=self.depth_K,
                K_color=self.color_K,
                align=self.align_depth,
            )
            depth_tensor = torch.from_numpy(d_proc)

        return (c_ts_ns * 1e-9, image, depth_tensor, intr)

    def _iter_frames(self):
        """
        Iterate all bag segments sequentially, yielding matched RGB-D frames.

        The step counter (for stride) is shared across bags so stride is
        consistent end-to-end.  depth_buf and pending are reset for each bag
        because bags are always split on a complete frame boundary — no
        cross-bag color/depth pairing is ever needed.
        """
        MAX_GAP_NS = int(self.MAX_SYNC_GAP_S * 1e9)
        step = 0  # absolute color-frame index across all bag segments

        for bagfile in self.bagfiles:
            depth_buf = deque(maxlen=8)
            pending = deque()

            def drain(force=False):
                while pending:
                    c_ts, c_raw = pending[0]
                    if not depth_buf:
                        if force:
                            pending.popleft()
                            yield c_ts, c_raw, None
                        return
                    nearest_dt, nearest_d = min(
                        ((abs(d_ts - c_ts), d) for d_ts, d in depth_buf),
                        key=lambda x: x[0],
                    )
                    if nearest_dt <= MAX_GAP_NS:
                        pending.popleft()
                        yield c_ts, c_raw, nearest_d
                    elif force:
                        pending.popleft()
                        yield c_ts, c_raw, None
                    else:
                        return

            with Reader(bagfile) as bag:
                color_conns, depth_conns = self._init_msgtypes(bag, bagfile)
                all_conns = color_conns + depth_conns

                for conn, ts_ns, rawdata in bag.messages(connections=all_conns):

                    if conn.topic == self.depth_topic:
                        try:
                            d = self._decode_depth_raw(rawdata)
                            if d is not None:
                                depth_buf.append((ts_ns, d))
                        except Exception as e:
                            print(f"[RosbagRGBDStream] depth error at t={ts_ns*1e-9:.3f}: {e}")
                        for c_ts, c_raw, dm in drain():
                            frame = self._build_frame(c_ts, c_raw, dm)
                            if frame is not None:
                                yield frame

                    else:  # color
                        tstamp = ts_ns * 1e-9
                        if tstamp < self.t0:
                            continue
                        if step % self.stride != 0:
                            step += 1
                            continue
                        step += 1
                        pending.append((ts_ns, rawdata))

                        for c_ts, c_raw, dm in drain():
                            frame = self._build_frame(c_ts, c_raw, dm)
                            if frame is not None:
                                yield frame

                        while pending and (ts_ns - pending[0][0]) > MAX_GAP_NS * 3:
                            c_ts, c_raw = pending.popleft()
                            frame = self._build_frame(c_ts, c_raw, None)
                            if frame is not None:
                                yield frame

                # End of this bag segment — flush remaining pending frames
                for c_ts, c_raw, dm in drain(force=True):
                    frame = self._build_frame(c_ts, c_raw, dm)
                    if frame is not None:
                        yield frame

    def __iter__(self):
        return self._iter_frames()

    def __len__(self):
        """Approximate frame count summed across all bag segments."""
        if self._approx_len is None:
            total = 0
            for bagfile in self.bagfiles:
                with Reader(bagfile) as bag:
                    color_conns, _ = self._init_msgtypes(bag, bagfile)
                    total += sum(c.msgcount for c in color_conns)
            self._approx_len = (total + self.stride - 1) // self.stride
        return self._approx_len

    def image_size(self):
        """Return [H, W] of preprocessed frames (reads first frame only, cached)."""
        if self._image_size is None:
            frame = next(self._iter_frames())
            self._image_size = [frame[1].shape[2], frame[1].shape[3]]
        return self._image_size

    def as_color_stream(self):
        """Return a RosbagColorStream over the same bag(s) (for droid.terminate())."""
        return RosbagColorStream(
            bagfile=self.bagfiles,
            color_topic=self.color_topic,
            intrinsics=self.color_K,
            target_area=self.target_area,
            stride=self.stride,
            t0=self.t0,
        )
