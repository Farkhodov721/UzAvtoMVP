#!/usr/bin/env python3
"""
Professional PPE Detection Dashboard
Clean, organized data display
"""

from ultralytics import YOLO
import cv2
import time
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "ppe_local.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.25

# Colors (BGR)
COLOR_SAFE = (0, 200, 0)
COLOR_UNSAFE = (0, 50, 200)
COLOR_BG_DARK = (20, 20, 20)
COLOR_BG_LIGHT = (40, 40, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (0, 255, 255)


# ============================================================
# PROFESSIONAL DETECTOR
# ============================================================

class ProfessionalDetector:
    def __init__(self, model_path, confidence=0.25):
        print("\n" + "=" * 60)
        print("   PROFESSIONAL PPE DETECTION SYSTEM")
        print("=" * 60 + "\n")

        print("🔄 Loading model...")
        self.model = YOLO(model_path)
        self.confidence = confidence

        print("✅ Model loaded!")
        print(f"📊 Classes: {list(self.model.names.values())}\n")

        # Statistics
        self.total_detections = 0
        self.detection_history = []

    def get_ppe_color(self, class_name):
        """Color coding for PPE"""
        name = class_name.lower()

        colors = {
            'helmet': (255, 0, 255),
            'hardhat': (255, 0, 255),
            'vest': (0, 255, 255),
            'jacket': (0, 255, 255),
            'glass': (0, 255, 0),
            'goggle': (0, 255, 0),
            'mask': (255, 255, 0),
            'glove': (255, 128, 0),
            'boot': (128, 0, 255),
            'person': (100, 150, 255),
        }

        for key, color in colors.items():
            if key in name:
                return color

        return (255, 255, 255)

    def draw_detection_boxes(self, frame, results):
        """Draw clean detection boxes"""
        detections = []
        boxes = results.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            color = self.get_ppe_color(class_name)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Label
            label = f"{class_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

            detections.append(class_name)

        return detections

    def draw_dashboard(self, frame, detections, fps, frame_count, elapsed_time):
        """Draw organized data dashboard"""
        h, w = frame.shape[:2]

        # ===== TOP BAR =====
        bar_h = 60

        if detections:
            bar_color = COLOR_SAFE
            status = f"ACTIVE: {len(detections)} items"
        else:
            bar_color = COLOR_UNSAFE
            status = "MONITORING"

        cv2.rectangle(frame, (0, 0), (w, bar_h), bar_color, -1)
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 2)

        # ===== RIGHT PANEL =====
        panel_w = 320
        panel_x = w - panel_w
        panel_y = bar_h + 20

        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y), (w - 20, h - 80),
                      COLOR_BG_DARK, -1)
        cv2.rectangle(frame, (panel_x, panel_y), (w - 20, h - 80),
                      COLOR_ACCENT, 2)

        # Panel content
        y = panel_y + 40
        line_h = 35

        # Title
        cv2.putText(frame, "SYSTEM STATUS", (panel_x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)

        y += line_h + 10
        cv2.line(frame, (panel_x + 20, y - 5), (w - 40, y - 5), COLOR_ACCENT, 1)

        # Data rows
        data = [
            ("FPS", f"{fps:.0f}"),
            ("Frames", f"{frame_count}"),
            ("Time", f"{elapsed_time:.0f}s"),
            ("Conf", f"{self.confidence * 100:.0f}%"),
        ]

        for label, value in data:
            # Label
            cv2.putText(frame, label, (panel_x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Value
            cv2.putText(frame, value, (panel_x + 180, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

            y += line_h

        # ===== DETECTION LIST =====
        if detections:
            y += 20
            cv2.line(frame, (panel_x + 20, y - 10), (w - 40, y - 10), COLOR_ACCENT, 1)

            cv2.putText(frame, "DETECTIONS", (panel_x + 20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)

            y += 50

            # Count by type
            counts = Counter(detections)

            for item, count in sorted(counts.items()):
                color = self.get_ppe_color(item)

                # Color dot
                cv2.circle(frame, (panel_x + 30, y - 5), 6, color, -1)
                cv2.circle(frame, (panel_x + 30, y - 5), 6, COLOR_TEXT, 1)

                # Item name
                cv2.putText(frame, item.title(), (panel_x + 50, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

                # Count
                cv2.putText(frame, f"x{count}", (panel_x + 240, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ACCENT, 2)

                y += 30

        # ===== BOTTOM BAR =====
        bottom_h = 50
        cv2.rectangle(frame, (0, h - bottom_h), (w, h), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, h - bottom_h), (w, h - bottom_h), COLOR_ACCENT, 2)

        # Bottom info
        cv2.putText(frame, "Q-Quit  S-Save  +/-Conf  SPACE-Pause",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.putText(frame, "LOCAL MODEL", (w - 180, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2)

    def process_frame(self, frame, fps, frame_count, elapsed_time):
        """Main processing pipeline"""
        # Detect
        results = self.model(frame, conf=self.confidence, verbose=False)[0]

        # Draw boxes
        detections = self.draw_detection_boxes(frame, results)

        # Draw dashboard
        self.draw_dashboard(frame, detections, fps, frame_count, elapsed_time)

        # Update stats
        self.total_detections += len(detections)

        return frame, len(detections)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    try:
        detector = ProfessionalDetector(MODEL_PATH, CONFIDENCE)
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("❌ Cannot open camera!\n")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("✅ Camera ready!")
    print("\n🎬 Starting...\n")

    frame_count = 0
    start_time = time.time()
    fps = 0
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                elapsed = time.time() - start_time

                # Calculate FPS
                if frame_count % 30 == 0:
                    fps = frame_count / elapsed

                # Process
                annotated, det_count = detector.process_frame(frame, fps, frame_count, elapsed)

                current_frame = annotated

            # Pause overlay
            if paused:
                h, w = current_frame.shape[:2]
                cv2.rectangle(current_frame, (w // 2 - 100, h // 2 - 40),
                              (w // 2 + 100, h // 2 + 40), COLOR_BG_DARK, -1)
                cv2.rectangle(current_frame, (w // 2 - 100, h // 2 - 40),
                              (w // 2 + 100, h // 2 + 40), COLOR_ACCENT, 3)
                cv2.putText(current_frame, "PAUSED", (w // 2 - 70, h // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT, 3)

            cv2.imshow('PPE Detection System', current_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            elif key == ord('s'):
                filename = f"ppe_{int(time.time())}.jpg"
                cv2.imwrite(filename, current_frame)
                print(f"📸 Saved: {filename}")

            elif key == ord('+') or key == ord('='):
                detector.confidence = min(0.95, detector.confidence + 0.05)
                print(f"⬆️  Confidence: {detector.confidence * 100:.0f}%")

            elif key == ord('-') or key == ord('_'):
                detector.confidence = max(0.05, detector.confidence - 0.05)
                print(f"⬇️  Confidence: {detector.confidence * 100:.0f}%")

            elif key == ord(' '):
                paused = not paused
                print(f"⏸️  {'PAUSED' if paused else 'RESUMED'}")

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed_total = time.time() - start_time

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Frames:              {frame_count}")
        print(f"Average FPS:         {fps:.1f}")
        print(f"Duration:            {elapsed_total:.1f}s")
        print(f"Total detections:    {detector.total_detections}")
        print(f"Avg per frame:       {detector.total_detections / frame_count:.2f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()