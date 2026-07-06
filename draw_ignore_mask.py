import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {args.frame}")

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    points = []
    window = "draw ignore mask"

    def render():
        view = frame.copy()
        overlay = view.copy()
        overlay[mask > 0] = (0, 0, 255)
        view = cv2.addWeighted(overlay, 0.35, view, 0.65, 0)
        for p in points:
            cv2.circle(view, p, 4, (0, 255, 255), -1)
        if len(points) > 1:
            cv2.polylines(view, [np.array(points)], False, (0, 255, 255), 2)
        cv2.putText(view, "left-click points | enter/right-click close | c clear | s save | q quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return view

    def close_polygon():
        nonlocal points
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
        points = []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            close_polygon()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)

    while True:
        cv2.imshow(window, render())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            close_polygon()
        elif key == ord("c"):
            mask[:] = 0
            points = []
        elif key == ord("s"):
            close_polygon()
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), mask)
            print(f"saved: {out}")
            break
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
