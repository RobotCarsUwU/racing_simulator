# from numpy import linspace, cos, sin, empty, float32, uint8, array, frombuffer, dot


from numpy cimport float32_t
cimport numpy as np
import numpy as np
import math
from PIL import Image
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)


cpdef raycast_fast_lookup(np.uint8_t[:, :] g, int n=30, int max_distance=500):
    """Version avec lookup table pour les calculs trigonométriques (Cython memoryview)"""
    cdef Py_ssize_t h = g.shape[0]
    cdef Py_ssize_t w = g.shape[1]
    cdef Py_ssize_t ox = w // 2
    cdef Py_ssize_t oy = h - 1

    cdef np.ndarray angles = np.linspace(-math.pi, math.pi, n, dtype=np.float32)
    cdef np.ndarray cos_table = np.cos(angles)
    cdef np.ndarray sin_table = np.sin(angles)
    cdef np.ndarray arr = np.empty(n, dtype=np.float32)
    cdef float32_t[:] distances = arr

    cdef int max_t = int(math.sqrt(w * w + h * h))
    cdef int i
    cdef float dx, dy, x, y, step, dx_step, dy_step, distance
    cdef int x_int, y_int
    cdef bint found

    for i in range(n):
        dx = cos_table[i]
        dy = sin_table[i]
        x = float(ox)
        y = float(oy)
        step = min(abs(1.0 / dx) if dx != 0 else float('inf'),
                  abs(1.0 / dy) if dy != 0 else float('inf'))
        step = min(step, 1.0)
        dx_step = dx * step
        dy_step = dy * step
        distance = 0.0
        found = False
        while distance < max_t:
            x_int = int(x)
            y_int = int(y)
            if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
                break
            if g[y_int, x_int] > 128:
                distances[i] = distance
                found = True
                break
            x += dx_step
            y += dy_step
            distance += step
        if not found:
            distances[i] = max_distance
    return distances

def preprocess_image_fast(image_path_or_array):
    """Prétraitement optimisé pour Python 3.6, version Cython"""
    # Use Python variable declarations for compatibility
    # g = None
    if isinstance(image_path_or_array, str):
        try:
            with Image.open(image_path_or_array) as img:
                if img.mode != 'L':
                    img = img.convert('L')
                g = np.frombuffer(img.tobytes(), dtype=np.uint8)
                g = g.reshape(img.size[1], img.size[0])
        except Exception:
            return None
    else:
        if hasattr(image_path_or_array, 'ndim') and image_path_or_array.ndim == 3:
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            g = np.dot(image_path_or_array[...,:3], weights).astype(np.uint8)
        else:
            if hasattr(image_path_or_array, 'dtype') and image_path_or_array.dtype != np.uint8:
                g = image_path_or_array.astype(np.uint8)
            else:
                g = image_path_or_array
    # Ensure array is writable for Cython memoryview
    arr = g if isinstance(g, np.ndarray) else np.asarray(g)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr

def raycast(image_path_or_array, n=30, max_distance=500):
    cdef np.uint8_t[:, :] g = preprocess_image_fast(image_path_or_array)
    if g is None:
        return np.array([[]])
    cdef np.uint8_t[:, :] g_view = g
    arr = np.asarray(raycast_fast_lookup(g_view, n, max_distance))
    return arr.reshape(1, -1)

if __name__ == "__main__":
    for _ in range(1000):
        raycast("../mask_generator/car_pictures/320_180/frame_00184.png", 50)
    print(raycast("../mask_generator/car_pictures/320_180/frame_00184.png", 50))