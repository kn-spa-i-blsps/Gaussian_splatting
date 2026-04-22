import argparse
from camera_pose import build_nadir_pose
from gaussian_renderer import render_gaussian
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    args = parser.parse_args()

    c2w = build_nadir_pose(args.x, args.y, args.z)

    img = render_gaussian(c2w)

    Image.fromarray(img).save("outputs/out.png")

    print("Saved outputs/out.png")

if __name__ == "__main__":
    main()
