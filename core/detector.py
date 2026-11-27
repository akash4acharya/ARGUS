import os
from typing import List, Tuple, Any
import numpy as np

# Standard COCO vehicle classes:
# 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASSES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK",
}


class YOLOv8VehicleDetector:
    """
    YOLOv8 wrapper specifically tuned for traffic surveillance and vehicle detection.
    Filters COCO classes to only detect cars, motorcycles, buses, and trucks.
    Formats outputs directly for consumption by the DeepSORT tracker.
    """

    def __init__(
        self,
        model_weights: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: List[int] = None,
        device: str = "cpu",
    ):
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or [2, 3, 5, 7]
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_weights)
        except Exception as e:
            print(f"[WARN] Could not initialize YOLOv8 model ({self.model_weights}): {e}")
            print("[INFO] Fallback detector mode active. Install ultralytics to enable full YOLOv8 inference.")
            self.model = None

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[List[float], float, str]]:
        """
        Runs object detection on a single BGR frame.

        Returns:
            List of DeepSORT formatted detections:
            [([left, top, width, height], confidence, class_name), ...]
        """
        if self.model is None or frame_bgr is None:
            return []

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                device=self.device,
                verbose=False,
            )

            detections: List[Tuple[List[float], float, str]] = []

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    # Coordinates in xyxy format
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    if self.model is not None and hasattr(self.model, "names") and isinstance(self.model.names, dict) and cls_id in self.model.names:
                        class_name = str(self.model.names[cls_id]).upper()
                    else:
                        class_name = COCO_VEHICLE_CLASSES.get(cls_id, "VEHICLE")
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)

                    # Skip tiny distant horizon specks to keep multi-lane HUD clean
                    if w < 28.0 or h < 24.0:
                        continue

                    # DeepSORT expects [left, top, width, height]
                    detections.append(([x1, y1, w, h], conf, class_name))

            return detections
        except Exception as err:
            print(f"[ERROR] Inference error in YOLOv8VehicleDetector: {err}")
            return []
