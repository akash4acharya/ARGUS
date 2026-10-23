import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    tire_point: Tuple[float, float]  # (u, v) = ((x1 + x2) / 2, y2)


class SimpleFallbackTracker:
    """
    Lightweight, dependency-free spatial IOU tracking fallback.
    Used if deep-sort-realtime is not installed or when running in low-resource environments.
    """

    def __init__(self, max_missed: int = 15, iou_threshold: float = 0.3):
        self.next_id = 1
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        # track_id -> {'bbox': [x1, y1, x2, y2], 'class': str, 'conf': float, 'missed': int, 'hits': int}
        self.tracks = {}

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections: List[Tuple[List[float], float, str]]) -> List[TrackedVehicle]:
        # Convert detections from [left, top, w, h] to [x1, y1, x2, y2]
        det_boxes = []
        for (ltwh, conf, cls_name) in detections:
            x1, y1, w, h = ltwh
            det_boxes.append(([x1, y1, x1 + w, y1 + h], conf, cls_name))

        matched_tracks = set()
        matched_dets = set()
        output_vehicles = []

        # Match existing tracks to detections by highest IOU
        for track_id, trk in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = -1
            for d_idx, (box, conf, cls_name) in enumerate(det_boxes):
                if d_idx in matched_dets:
                    continue
                iou_val = self._iou(trk['bbox'], box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = d_idx

            if best_iou >= self.iou_threshold and best_det_idx >= 0:
                box, conf, cls_name = det_boxes[best_det_idx]
                trk['bbox'] = box
                trk['class'] = cls_name
                trk['conf'] = conf
                trk['missed'] = 0
                trk['hits'] += 1
                matched_tracks.add(track_id)
                matched_dets.add(best_det_idx)
            else:
                trk['missed'] += 1

        # Delete lost tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['missed'] > self.max_missed:
                del self.tracks[track_id]

        # Create new tracks for unmatched detections
        for d_idx, (box, conf, cls_name) in enumerate(det_boxes):
            if d_idx not in matched_dets:
                self.tracks[self.next_id] = {
                    'bbox': box,
                    'class': cls_name,
                    'conf': conf,
                    'missed': 0,
                    'hits': 1,
                }
                self.next_id += 1

        # Emit confirmed tracks (hits >= 2)
        for track_id, trk in self.tracks.items():
            if trk['hits'] >= 2 and trk['missed'] == 0:
                x1, y1, x2, y2 = trk['bbox']
                # Skill 2: Bottom-center tire contact point
                u = (x1 + x2) / 2.0
                v = float(y2)
                output_vehicles.append(
                    TrackedVehicle(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        class_name=trk['class'],
                        confidence=trk['conf'],
                        tire_point=(u, v),
                    )
                )

        return output_vehicles


class DeepSORTVehicleTracker:
    """
    Multi-Object Tracker combining appearance embeddings and Kalman filtering.
    Maintains persistent track IDs for vehicles across CCTV camera feeds.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        nms_max_overlap: float = 1.0,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.nms_max_overlap = nms_max_overlap
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.deepsort = None
        self.fallback_tracker = SimpleFallbackTracker(max_missed=max_age)

        self._init_tracker()

    def _init_tracker(self) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.deepsort = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                nms_max_overlap=self.nms_max_overlap,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                override_track_class=None,
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=False,
            )
        except Exception as e:
            print(f"[WARN] DeepSORT failed to initialize ({e}). Using robust fallback tracker.")
            self.deepsort = None

    def update(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame_bgr: np.ndarray,
    ) -> List[TrackedVehicle]:
        """
        Updates multi-object trajectories with new frame detections.

        Args:
            detections: List of [([left, top, w, h], confidence, class_name)]
            frame_bgr: Current BGR image array

        Returns:
            List of TrackedVehicle objects
        """
        if self.deepsort is None or frame_bgr is None:
            return self.fallback_tracker.update(detections)

        try:
            raw_tracks = self.deepsort.update_tracks(detections, frame=frame_bgr)
            tracked_vehicles: List[TrackedVehicle] = []

            for track in raw_tracks:
                if not track.is_confirmed():
                    continue

                track_id_raw = track.track_id
                try:
                    track_id = int(track_id_raw)
                except (ValueError, TypeError):
                    track_id = hash(track_id_raw) % 100000

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])

                # Skill 2: Bottom-center tire contact point
                u = (x1 + x2) / 2.0
                v = y2

                class_name = track.get_det_class() or "CAR"
                conf = float(track.get_det_conf() or 0.85)

                tracked_vehicles.append(
                    TrackedVehicle(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        class_name=str(class_name).upper(),
                        confidence=conf,
                        tire_point=(u, v),
                    )
                )

            return tracked_vehicles
        except Exception as err:
            print(f"[WARN] Error updating DeepSORT tracks ({err}), using fallback.")
            return self.fallback_tracker.update(detections)
