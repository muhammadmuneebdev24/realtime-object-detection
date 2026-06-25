"""
Simple IoU-based tracker — no neural net, no external deps, instant speed.
Matches detections frame-to-frame by overlap (IoU). Tracks disappear after
max_age missed frames, so no ghost boxes.
"""

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class Track:
    def __init__(self, tid, box, label):
        self.track_id = tid
        self.box = box        # (x1, y1, x2, y2)
        self.label = label
        self.missed = 0

    def to_ltrb(self):
        return self.box

    def get_det_class(self):
        return self.label

    def is_confirmed(self):
        return True


class SimpleTracker:
    def __init__(self, max_age=3, iou_thresh=0.3):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self._tracks = []
        self._next_id = 1

    def update(self, detections):
        """Match detections to existing tracks by IoU."""
        matched_track_ids = set()
        matched_det_ids = set()

        # Match each detection to the best-overlapping existing track
        for di, det in enumerate(detections):
            x1, y1, x2, y2, conf, label = det
            best_iou, best_ti = 0.0, -1
            for ti, trk in enumerate(self._tracks):
                if ti in matched_track_ids:
                    continue
                overlap = iou((x1, y1, x2, y2), trk.box)
                if overlap > best_iou:
                    best_iou, best_ti = overlap, ti
            if best_iou >= self.iou_thresh:
                self._tracks[best_ti].box = (x1, y1, x2, y2)
                self._tracks[best_ti].label = label
                self._tracks[best_ti].missed = 0
                matched_track_ids.add(best_ti)
                matched_det_ids.add(di)

        # New tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                x1, y1, x2, y2, conf, label = det
                self._tracks.append(Track(self._next_id, (x1, y1, x2, y2), label))
                self._next_id += 1

        # Age out unmatched tracks
        for ti, trk in enumerate(self._tracks):
            if ti not in matched_track_ids:
                trk.missed += 1

        # Remove old tracks
        self._tracks = [t for t in self._tracks if t.missed <= self.max_age]

        return list(self._tracks)


# Global tracker instance
tracker = SimpleTracker(max_age=3, iou_thresh=0.3)


def track_objects(detections, frame):
    return tracker.update(detections)