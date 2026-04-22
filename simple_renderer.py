import numpy as np

def render_simple(c2w, width=512, height=512):
    pos = c2w[:3, 3]

    img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            img[y, x, 0] = int((pos[0] * 50 + x) % 255)
            img[y, x, 1] = int((pos[1] * 50 + y) % 255)
            img[y, x, 2] = int((pos[2] * 50) % 255)

    return img
