import cv2
import numpy as np
import glob
import os

# ── Settings ──────────────────────────────────────────────────────────────────
VIDEO_FILE   = 'chessboard.mov'   # <-- change to your video filename
BOARD_SIZE   = (6, 9)             # inner corners (cols, rows) — must match your board
SQUARE_SIZE  = 1.0                # real-world square size (any unit, e.g. cm)
FRAME_SKIP   = 15                 # sample every N frames from the video
OUTPUT_FILE  = 'calibration.npz'  # where to save the results
# ──────────────────────────────────────────────────────────────────────────────

def calibrate(video_path, board_size, square_size, frame_skip):
    # Prepare object points: (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened: {total_frames} frames total, sampling every {frame_skip} frames")

    frame_idx   = 0
    found_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_cb, corners = cv2.findChessboardCorners(gray, board_size, None)

            if ret_cb:
                # Refine corner positions to sub-pixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                obj_points.append(objp)
                img_points.append(corners2)
                found_count += 1

                # Draw and show detected corners
                cv2.drawChessboardCorners(frame, board_size, corners2, ret_cb)
                print(f"  Frame {frame_idx:5d}: chessboard found! (total: {found_count})")
            else:
                print(f"  Frame {frame_idx:5d}: not found")

            cv2.imshow('Calibration frames', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped early by user.")
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if found_count < 10:
        print(f"\nWarning: only {found_count} valid frames found. Try reducing FRAME_SKIP or recapturing.")
        if found_count == 0:
            return

    print(f"\nRunning calibration with {found_count} frames...")
    h, w = gray.shape
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    # ── Results ───────────────────────────────────────────────────────────────
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Compute RMSE (re-projection error)
    total_error = 0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        total_error += cv2.norm(img_points[i], proj, cv2.NORM_L2) ** 2
    rmse = np.sqrt(total_error / sum(len(p) for p in img_points))

    print("\n── Camera Calibration Results ───────────────────────────────────")
    print(f"  Image size : {w} x {h} px")
    print(f"  fx         : {fx:.4f}")
    print(f"  fy         : {fy:.4f}")
    print(f"  cx         : {cx:.4f}")
    print(f"  cy         : {cy:.4f}")
    print(f"  Distortion : {dist.ravel()}")
    print(f"  RMSE       : {rmse:.4f} px")
    print("─────────────────────────────────────────────────────────────────")
    print(f"\nSaved calibration to '{OUTPUT_FILE}'")

    np.savez(OUTPUT_FILE, K=K, dist=dist, image_size=(w, h))

    # ── Copy results to README-friendly format ────────────────────────────────
    readme_block = f"""## Camera Calibration Results

| Parameter | Value |
|-----------|-------|
| fx | {fx:.4f} |
| fy | {fy:.4f} |
| cx | {cx:.4f} |
| cy | {cy:.4f} |
| k1 | {dist[0][0]:.6f} |
| k2 | {dist[0][1]:.6f} |
| p1 | {dist[0][2]:.6f} |
| p2 | {dist[0][3]:.6f} |
| k3 | {dist[0][4]:.6f} |
| RMSE | {rmse:.4f} px |
"""
    with open('calibration_readme.md', 'w') as f:
        f.write(readme_block)
    print("README-ready table saved to 'calibration_readme.md'")

if __name__ == '__main__':
    if not os.path.exists(VIDEO_FILE):
        print(f"Video file '{VIDEO_FILE}' not found.")
        print("Please set VIDEO_FILE at the top of the script to your video's filename.")
    else:
        calibrate(VIDEO_FILE, BOARD_SIZE, SQUARE_SIZE, FRAME_SKIP)