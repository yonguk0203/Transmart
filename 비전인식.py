import time
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# === ROS2 Control ===
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


MODEL_PATH = "/home/hong/best.pt"
CLASS_IDS   = [0, 1]
IMG_SIZE    = 640
CONF_THRES  = 0.35
IOU_THRES   = 0.30
UNLOCK_MISS_FRAMES = 10
WINDOW_NAME = "YOLOv8 Hand Lock (D435 USB)"
MEDIAN_K    = 2


def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    if inter <= 0:
        return 0.0
    areaA = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    areaB = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    return inter / (areaA + areaB - inter + 1e-9)


def median_depth_m(depth_img, cx, cy, k=2, depth_scale=0.001):
    h, w = depth_img.shape[:2]
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    x0 = max(0, cx - k); x1 = min(w, cx + k + 1)
    y0 = max(0, cy - k); y1 = min(h, cy + k + 1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    mm = float(np.median(patch))
    if mm <= 0:
        return None
    return mm * depth_scale


def main():
    model = YOLO(MODEL_PATH)

    # === RealSense ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # === ROS2 Init ===
    rclpy.init()
    node = Node("yolo_ctrl_node")
    cmd_pub = node.create_publisher(Twist, "/cmd_vel", 10)

    stop_msg = Twist()  # 정지 명령
    move_msg = Twist()
    move_msg.linear.x = 0.15   # 전진 속도

    locked = False
    lock_box = None
    lock_cls = None
    lock_conf = None
    miss_frames = 0

    prev_t = time.time()
    fps = 0.0

    mode = "RUN"
    show_run = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            frame = np.asanyarray(color_frame.get_data())
            disp = frame.copy()

            dist_m = None
            det_boxes, det_confs, det_clss = [], [], []

            if mode == "RUN":
                # 움직이는 상태면 계속 속도 전송
                if show_run:
                    cmd_pub.publish(move_msg)

                yres = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
                if len(yres) > 0:
                    r0 = yres[0]
                    if r0.boxes is not None and len(r0.boxes) > 0:
                        xyxy = r0.boxes.xyxy.cpu().numpy()
                        conf = r0.boxes.conf.cpu().numpy()
                        cls  = r0.boxes.cls.cpu().numpy().astype(int)
                        for b, c, k in zip(xyxy, conf, cls):
                            if (CLASS_IDS is None) or (k in CLASS_IDS):
                                det_boxes.append(b.tolist())
                                det_confs.append(float(c))
                                det_clss.append(int(k))

                if not locked:
                    if det_boxes:
                        locked     = True
                        lock_box   = det_boxes[0]
                        lock_cls   = det_clss[0]
                        lock_conf  = det_confs[0]
                        miss_frames = 0
                else:
                    best_iou, best_idx = 0.0, -1
                    for i, b in enumerate(det_boxes):
                        iou = iou_xyxy(lock_box, b)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    if best_iou >= IOU_THRES and best_idx >= 0:
                        lock_box   = det_boxes[best_idx]
                        lock_cls   = det_clss[best_idx]
                        lock_conf  = det_confs[best_idx]
                        miss_frames = 0
                    else:
                        miss_frames += 1
                        if miss_frames > UNLOCK_MISS_FRAMES:
                            locked = False
                            lock_box = None
                            lock_cls = None
                            lock_conf = None
                            miss_frames = 0

                for b, c, k in zip(det_boxes, det_confs, det_clss):
                    x1,y1,x2,y2 = map(int, b)
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 1)
                    cv2.putText(disp, f"{k}:{c:.2f}", (x1, max(0,y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                if locked and lock_box is not None:
                    x1,y1,x2,y2 = map(int, lock_box)
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 3)
                    cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                    dist_m = median_depth_m(depth_img, cx, cy, k=MEDIAN_K, depth_scale=depth_scale)

                    label = f"LOCK cls:{lock_cls} conf:{(lock_conf if lock_conf else 0):.2f}"
                    if dist_m is not None:
                        label += f"  dist:{dist_m:.2f}m"
                    cv2.putText(disp, label, (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.circle(disp, (cx, cy), 4, (0,0,255), -1)

                # ========================
                # 🚨 STOP Trigger
                # ========================
                if dist_m is not None and dist_m < 1.0:
                    mode = "STOP"
                    cmd_pub.publish(stop_msg)

            if mode == "STOP":
                cv2.putText(disp, "STOP", (disp.shape[1]//2 - 120, disp.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 8)
                cmd_pub.publish(stop_msg)

            if show_run:
                cv2.putText(disp, "RUN", (disp.shape[1]//2 - 100, disp.shape[0]//2 + 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 8)

            now = time.time()
            dt = now - prev_t
            if dt > 0:
                fps = 1.0 / dt
            prev_t = now
            cv2.putText(disp, f"MODE:{mode}  FPS:{fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cmd_pub.publish(stop_msg)
                break
            elif key == ord('1'):
                mode = "RUN"
                show_run = True
            elif key == ord('0'):
                show_run = False

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
