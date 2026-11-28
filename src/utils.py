import cv2
import numpy as np
from data_types import ArucoTag, State


def get_aruco_tag(image: np.ndarray, tag_ids: list[int]) -> list[ArucoTag] | None:
    """
    Given a aruco tag_id, return information on the tag's state (position, heading).
    Useful to get information on Robot's actual position.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    tags, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None

    target_set = set(tag_ids)

    target_map = {int(tag_id): tag for tag_id, tag in zip(ids.flatten(), tags, strict=False) if int(tag_id) in target_set}

    output: list[ArucoTag] = []

    for tid, tag in target_map.items():
        pts = tag[0]
        tl, tr, br, bl = pts
        centroid = pts.mean(axis=0)

        # Forward vector of the tag points from bottom-left to top-left
        dx = float(tl[0] - bl[0])
        dy = float(tl[1] - bl[1])

        # 0 deg at North (dy < 0, dx = 0), +90 deg at East (dx > 0, dy = 0), (-180, 180]
        heading_rad = np.arctan2(dx, -dy)
        heading_deg = float(np.degrees(heading_rad))

        output.append(
            ArucoTag(
                tid=tid,
                top_left_corner_pos=(float(tl[0]), float(tl[1])),
                top_right_corner_pos=(float(tr[0]), float(tr[1])),
                bottom_left_corner_pos=(float(bl[0]), float(bl[1])),
                bottom_right_corner_pos=(float(br[0]), float(br[1])),
                state=State(
                    position=(float(centroid[0]), float(centroid[1])),
                    heading=heading_deg,
                ),
            )
        )

    return output


def angle_clamp(a: float) -> float:
    """
    Given an angle a, make sure that it is properly formatted to the angle range we expect
    """
    if a > 180.0:
        return (a % 360.0) - 360.0
    elif a <= -180:
        return (a % -360.0) + 360.0
    else:
        return a


def angle_diff(a: float, b: float) -> float:
    """
    Find the angle from a to b (degrees), where clockwise is positive.
    Wrapped to stay in the range (-180, 180]
    """
    delta = b - a
    return angle_clamp(delta)
