# src/dust_simulator.py
import numpy as np
from scipy.ndimage import gaussian_filter

class DustSimulator:
    def __init__(self, room_width_cm, room_height_cm, room_depth_cm, voxel_size_cm=10):
        self.width = room_width_cm
        self.height = room_height_cm
        self.depth = room_depth_cm
        self.voxel_size = voxel_size_cm

        self.nx = int(self.width / voxel_size_cm) + 1
        self.ny = int(self.height / voxel_size_cm) + 1
        self.nz = int(self.depth / voxel_size_cm) + 1

        self.grid = np.zeros((self.nx, self.ny, self.nz))
        self.furniture = []  # (x, y, z, w, h, d, type)

    # ✅ Object-based emission
    def get_emission(self, obj_type):
        obj_type = obj_type.lower()
        if "bed" in obj_type:
            return 2.5
        elif "curtain" in obj_type:
            return 3.0
        elif "carpet" in obj_type:
            return 3.5
        elif "sofa" in obj_type:
            return 2.8
        elif "plant" in obj_type:
            return 1.2
        elif "table" in obj_type:
            return 1.0
        else:
            return 1.5

    def add_furniture(self, x_cm, y_cm, z_cm, width_cm, height_cm, depth_cm, obj_type="generic"):
        self.furniture.append((x_cm, y_cm, z_cm, width_cm, height_cm, depth_cm, obj_type))

        xi = int(x_cm / self.voxel_size)
        yi = int(y_cm / self.voxel_size)
        zi = int(z_cm / self.voxel_size)

        wi = int(width_cm / self.voxel_size)
        hi = int(height_cm / self.voxel_size)
        di = int(depth_cm / self.voxel_size)

        self.grid[xi:xi+wi, yi:yi+hi, zi:zi+di] = 1

    def simulate_dust(self, fan_speed=50, window_open=False, humidity=50, aqi=100):
        dust_source = np.zeros_like(self.grid)

        # ✅ Apply object-based emission
        for (x, y, z, w, h, d, obj_type) in self.furniture:
            xi = int(x / self.voxel_size)
            yi = int(y / self.voxel_size)
            zi = int(z / self.voxel_size)

            wi = int(w / self.voxel_size)
            hi = int(h / self.voxel_size)
            di = int(d / self.voxel_size)

            emission = self.get_emission(obj_type)

            dust_source[xi:xi+wi, yi:yi+hi, zi:zi+di] += emission

        # ✅ Create coordinate grid (vectorized airflow)
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        z = np.arange(self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        airflow = np.ones_like(self.grid) * 0.5

        # ✅ Fan effect (center-based airflow)
        if fan_speed > 0:
            cx, cy, cz = self.nx//2, self.ny//3, self.nz//2
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
            airflow += (fan_speed / 100.0) * np.exp(-dist / 10)

        # ✅ Window effect (directional airflow from one side)
        if window_open:
            airflow += (X / self.nx) * 0.5  # airflow increases near window wall

        # ✅ Environmental factors
        humidity_factor = 1 + (humidity - 50) / 100
        aqi_factor = aqi / 100

        # ✅ Core dust equation
        dust = dust_source / (airflow + 0.1)
        dust *= humidity_factor * aqi_factor

        # ✅ Gravity effect (dust settles downward)
        gravity = 1 - (Y / self.ny) * 0.4
        dust *= gravity

        # ✅ Diffusion (smooth spread)
        dust = gaussian_filter(dust, sigma=1)

        # ✅ Normalize
        if dust.max() > 0:
            dust = dust / dust.max()

        return dust

    def classify_risk(self, dust_concentration):
        low = np.percentile(dust_concentration, 33)
        high = np.percentile(dust_concentration, 66)

        risk = np.zeros_like(dust_concentration, dtype=int)
        risk[dust_concentration <= low] = 1
        risk[(dust_concentration > low) & (dust_concentration <= high)] = 2
        risk[dust_concentration > high] = 3

        return risk