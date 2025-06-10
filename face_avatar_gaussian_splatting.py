"""Simple face avatar using 3D Gaussian splatting without external dependencies."""

import math
import random

# ---- Utility functions ----

def generate_sphere_points(num_points):
    """Generate points on a unit sphere."""
    vertices = []
    for _ in range(num_points):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        vertices.append((x, y, z))
    return vertices


def uv_map(vertex):
    x, y, z = vertex
    u = 0.5 + math.atan2(z, x) / (2 * math.pi)
    v = 0.5 - math.asin(y) / math.pi
    return (u, v)


def regress_gaussian(uv):
    u, v = uv
    r, g, b = u, v, 0.5
    scale = 0.02 + 0.04 * (u ** 2)
    opacity = 0.8
    return (r, g, b, scale, opacity)


def project(vertex, camera_pos, look_at, image_size=256, fov=60):
    cx, cy, cz = camera_pos
    lx, ly, lz = look_at
    fx, fy, fz = lx - cx, ly - cy, lz - cz
    flen = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / flen, fy / flen, fz / flen
    up = (0.0, 1.0, 0.0)
    rx = fy * up[2] - fz * up[1]
    ry = fz * up[0] - fx * up[2]
    rz = fx * up[1] - fy * up[0]
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rlen, ry / rlen, rz / rlen
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    # transform
    x, y, z = vertex
    tx = rx * (x - cx) + ry * (y - cy) + rz * (z - cz)
    ty = ux * (x - cx) + uy * (y - cy) + uz * (z - cz)
    tz = -(fx * (x - cx) + fy * (y - cy) + fz * (z - cz))
    f = 1.0 / math.tan(math.radians(fov) / 2)
    if tz == 0:
        tz = 1e-6
    px = f * tx / tz
    py = f * ty / tz
    ix = int((px + 1) * image_size / 2)
    iy = int((1 - py) * image_size / 2)
    return ix, iy, tz


def render(vertices, gaussians, camera_pos, look_at, image_size=256):
    img = [[[0.0, 0.0, 0.0] for _ in range(image_size)] for _ in range(image_size)]
    depth = [[1e9 for _ in range(image_size)] for _ in range(image_size)]
    for v, g in zip(vertices, gaussians):
        ix, iy, z = project(v, camera_pos, look_at, image_size)
        if 0 <= ix < image_size and 0 <= iy < image_size:
            r, g_col, b, scale, opacity = g
            radius = max(1, int(scale * image_size))
            for i in range(max(0, iy - radius), min(image_size, iy + radius)):
                for j in range(max(0, ix - radius), min(image_size, ix + radius)):
                    d = math.sqrt((i - iy) ** 2 + (j - ix) ** 2)
                    if d < radius and z < depth[i][j]:
                        depth[i][j] = z
                        old = img[i][j]
                        img[i][j] = [
                            (1 - opacity) * old[0] + opacity * r,
                            (1 - opacity) * old[1] + opacity * g_col,
                            (1 - opacity) * old[2] + opacity * b,
                        ]
    return img


def save_ppm(image, path):
    h = len(image)
    w = len(image[0]) if h > 0 else 0
    with open(path, 'w') as f:
        f.write('P3\n{} {}\n255\n'.format(w, h))
        for row in image:
            for r, g, b in row:
                f.write('{} {} {} '.format(int(r * 255), int(g * 255), int(b * 255)))
            f.write('\n')


if __name__ == '__main__':
    vertices = generate_sphere_points(500)
    uvs = [uv_map(v) for v in vertices]
    gaussians = [regress_gaussian(uv) for uv in uvs]
    camera = (0.0, 0.0, 3.0)
    look = (0.0, 0.0, 0.0)
    img = render(vertices, gaussians, camera, look)
    save_ppm(img, 'render.ppm')
    print('Rendered image saved to render.ppm')
