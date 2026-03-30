import cv2
import numpy as np

# ── Settings ──────────────────────────────────────────────────────────────────
CALIBRATION_FILE = 'calibration.npz'  # output from camera_calibration.py
INPUT_SOURCE     = 'chessboard.mov'                  # 0 = webcam, or e.g. 'chessboard.mov'
SAVE_COMPARISON  = True               # save a before/after image for README
# ──────────────────────────────────────────────────────────────────────────────

def load_calibration(path):
    data = np.load(path)
    K    = data['K']
    dist = data['dist']
    w, h = data['image_size']
    print(f"Calibration loaded from '{path}'")
    print(f"  K    = \n{K}")
    print(f"  dist = {dist.ravel()}")
    return K, dist, int(w), int(h)

def run(calibration_file, source, save_comparison):
    K, dist, cal_w, cal_h = load_calibration(calibration_file)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source '{source}'")
        return

    # Compute the optimal new camera matrix once
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from source.")
        return

    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)

    saved = False
    print("\nShowing undistorted feed. Press 's' to save a comparison image, 'q' to quit.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply distortion correction
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Crop to valid region (optional — remove black borders)
        x, y, rw, rh = roi
        cropped = undistorted[y:y+rh, x:x+rw]

        # Show side by side
        # Resize both to same height for display
        display_h = 480
        scale_orig = display_h / frame.shape[0]
        scale_crop = display_h / cropped.shape[0]
        orig_disp  = cv2.resize(frame,     (int(frame.shape[1]  * scale_orig), display_h))
        crop_disp  = cv2.resize(cropped,   (int(cropped.shape[1] * scale_crop), display_h))

        # Add labels
        cv2.putText(orig_disp, 'Original (distorted)',   (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(crop_disp, 'Corrected (undistorted)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        comparison = np.hstack([orig_disp, crop_disp])
        cv2.imshow('Distortion Correction  |  s = save  q = quit', comparison)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') or (save_comparison and not saved):
            cv2.imwrite('distortion_comparison.jpg', comparison)
            print("Saved comparison image to 'distortion_comparison.jpg'")
            saved = True

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    run(CALIBRATION_FILE, INPUT_SOURCE, SAVE_COMPARISON)