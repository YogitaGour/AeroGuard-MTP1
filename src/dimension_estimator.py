class DimensionEstimator:
    def __init__(self, scale_cm_per_pixel):
        self.scale = scale_cm_per_pixel

    def estimate(self, detections):
        """detections from TemporalTracker (with width_px, height_px)"""
        for d in detections:
            d['width_cm'] = d['width_px'] * self.scale
            d['height_cm'] = d['height_px'] * self.scale
        return detections