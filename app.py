import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template_string, Response, request, jsonify

# ==========================================
# CONFIGURATION
# ==========================================
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "sb284160@gmail.com"
SMTP_PASS = "tpxb fhwo iooq pyvk"
TO_EMAIL = "sb284160@gmail.com"
SUBJECT = "Smart Parking System — Live Dashboard Link"
EMAIL_ENABLED = True

CAMERA_SOURCE = 0
SPOTS_FILE = "parking_spots.json"


# ==========================================
# EMAIL FUNCTION
# ==========================================
def send_email_notification(link):
    if not EMAIL_ENABLED:
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = TO_EMAIL
        msg['Subject'] = SUBJECT
        body = f"The Smart Parking System has started.\nView the live dashboard here:\n{link}"
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, TO_EMAIL, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {TO_EMAIL}")
    except Exception as e:
        print(f"Failed to send email: {e}")


# ==========================================
# GLOBAL STATE
# ==========================================
app = Flask(__name__)
model = YOLO('yolov8n.pt')

output_frame = None
lock = threading.Lock()
parking_spots = {}
is_calibrated = False
is_running = False
live_status = {}
stats = {"total": 0, "occupied": 0, "available": 0, "occupancy_rate": 0}


def load_and_clean_calibration():
    global parking_spots, is_calibrated, live_status, stats
    if os.path.exists(SPOTS_FILE):
        try:
            with open(SPOTS_FILE, 'r') as f:
                raw_data = json.load(f)
            clean_data = {}
            for spot_id, points in raw_data.items():
                if len(points) > 0 and isinstance(points[0], dict):
                    clean_data[spot_id] = [[pt['x'], pt['y']] for pt in points]
                else:
                    clean_data[spot_id] = points
            parking_spots = clean_data
            is_calibrated = True
            for spot_id in parking_spots.keys():
                live_status[spot_id] = "AVAILABLE"
            stats["total"] = len(parking_spots)
            stats["available"] = len(parking_spots)
            print(f"Successfully loaded and validated {len(parking_spots)} spots.")
            with open(SPOTS_FILE, 'w') as f:
                json.dump(parking_spots, f)
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            parking_spots = {}
            is_calibrated = False


load_and_clean_calibration()


# ==========================================
# VIDEO PROCESSING ENGINE
# ==========================================
def process_video():
    global output_frame, parking_spots, is_calibrated, is_running, live_status, stats
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        if not is_calibrated:
            cv2.putText(frame, "CALIBRATION MODE", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif is_calibrated and not is_running:
            cv2.putText(frame, "CLICK START PROCESS", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            for spot_id, poly_pts in parking_spots.items():
                if len(poly_pts) != 4: continue
                poly = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [poly], True, (0, 255, 0), 3)
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, f"{spot_id}", (cX - 15, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif is_running:
            results = model(frame, classes=[2, 5, 7], verbose=False)[0]
            boxes = results.boxes
            detected_vehicles = []

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                if conf > 0.4:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detected_vehicles.append((cx, cy))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            occupied_count = 0
            total_spots = len(parking_spots)

            for spot_id, poly_pts in parking_spots.items():
                if len(poly_pts) != 4: continue
                poly = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
                is_occupied = False
                for (cx, cy) in detected_vehicles:
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        is_occupied = True
                        break

                if is_occupied:
                    color = (0, 0, 255)
                    occupied_count += 1
                    live_status[spot_id] = "OCCUPIED"
                else:
                    color = (0, 255, 0)
                    live_status[spot_id] = "AVAILABLE"

                cv2.polylines(frame, [poly], True, color, 3)
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, f"{spot_id}", (cX - 15, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            stats["total"] = total_spots
            stats["occupied"] = occupied_count
            stats["available"] = total_spots - occupied_count
            stats["occupancy_rate"] = int((occupied_count / total_spots) * 100) if total_spots > 0 else 0

        with lock:
            output_frame = frame.copy()

    cap.release()


# ==========================================
# FLASK WEB SERVER & HTML/JS
# ==========================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, is_calibrated=is_calibrated, is_running=is_running)


@app.route('/video_feed')
def video_feed():
    def generate():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    time.sleep(0.1)
                    continue
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
                if not flag: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_status')
def get_status():
    global live_status, is_running, stats
    return jsonify({
        "is_running": is_running,
        "status": live_status,
        "stats": stats
    })


@app.route('/save_calibration', methods=['POST'])
def save_calibration():
    global parking_spots, is_calibrated, is_running, live_status, stats
    data = request.json
    formatted_spots = {}
    try:
        for spot_id, points in data.items():
            formatted_spots[spot_id] = [[pt['x'], pt['y']] for pt in points]
        with open(SPOTS_FILE, 'w') as f:
            json.dump(formatted_spots, f)
        parking_spots = formatted_spots
        live_status = {spot_id: "AVAILABLE" for spot_id in parking_spots.keys()}
        stats["total"] = len(parking_spots)
        stats["available"] = len(parking_spots)
        is_calibrated = True
        is_running = False
        return jsonify(
            {"status": "success", "message": "Calibration saved successfully! Click 'Start Process' to begin."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/reset_calibration', methods=['POST'])
def reset_calibration():
    global is_calibrated, is_running, live_status, stats
    is_calibrated = False
    is_running = False
    live_status = {}
    stats = {"total": 0, "occupied": 0, "available": 0, "occupancy_rate": 0}
    if os.path.exists(SPOTS_FILE):
        os.remove(SPOTS_FILE)
    return jsonify({"status": "success", "message": "Calibration reset. Refresh to redraw."})


@app.route('/start_process', methods=['POST'])
def start_process():
    global is_running
    is_running = True
    return jsonify({"status": "success", "message": "Process started!"})


# ==========================================
# SINGLE SCREEN HTML TEMPLATE
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Parking | Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #05080f;
            --bg-panel: rgba(16, 24, 39, 0.7);
            --border-glow: rgba(88, 166, 255, 0.3);
            --accent-blue: #58a6ff;
            --accent-green: #23d160;
            --accent-red: #f85149;
            --accent-orange: #f0883e;
            --text-main: #e6edf3;
            --text-muted: #8b949e;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        /* Animated Background Grid */
        .bg-grid {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background-image: 
                linear-gradient(rgba(88, 166, 255, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(88, 166, 255, 0.05) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: -1;
            animation: gridMove 20s linear infinite;
        }
        @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 30px;
            background: rgba(13, 17, 23, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border-glow);
            z-index: 100;
            flex-shrink: 0;
        }
        .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.4rem;
            font-weight: 900;
            letter-spacing: 3px;
            color: var(--text-main);
            text-shadow: 0 0 10px var(--accent-blue);
        }
        .logo span { color: var(--accent-blue); }
        .system-status {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85rem;
            color: var(--accent-green);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .live-dot {
            width: 10px; height: 10px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(35, 209, 96, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(35, 209, 96, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(35, 209, 96, 0); }
        }

        /* Main Dashboard - Flex to fit screen */
        .dashboard {
            display: flex;
            flex: 1;
            padding: 15px 30px;
            gap: 20px;
            overflow: hidden;
            height: calc(100vh - 60px);
        }

        /* Left Side - Video */
        .video-section {
            flex: 2;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .video-container {
            flex: 1;
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .video-container canvas {
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            height: 100%;
            cursor: crosshair;
        }

        /* Right Side - Status Panel */
        .status-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow: hidden;
        }

        /* Big Availability Display */
        .big-availability {
            background: var(--bg-panel);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
            flex-shrink: 0;
        }
        .big-availability .label {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            color: var(--text-muted);
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .big-availability .value {
            font-family: 'Orbitron', sans-serif;
            font-size: 4rem;
            font-weight: 900;
            color: var(--accent-green);
            text-shadow: 0 0 20px rgba(35, 209, 96, 0.5);
            line-height: 1;
        }
        .big-availability .sub {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 5px;
        }
        .big-availability .sub span { color: var(--accent-red); font-weight: bold; }

        /* Slot Grid */
        .slot-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            flex: 1;
            overflow-y: auto;
            padding-right: 5px;
        }
        /* Custom Scrollbar for slot grid */
        .slot-grid::-webkit-scrollbar { width: 4px; }
        .slot-grid::-webkit-scrollbar-track { background: transparent; }
        .slot-grid::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 4px; }

        .slot-card {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            min-height: 80px;
        }
        .slot-card .slot-id {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 4px;
            z-index: 1;
        }
        .slot-card .slot-status {
            font-size: 0.65rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
            z-index: 1;
        }

        .available { border-color: rgba(35, 209, 96, 0.4); }
        .available .slot-id { color: var(--accent-green); }
        .available .slot-status { background: rgba(35, 209, 96, 0.15); color: var(--accent-green); border: 1px solid rgba(35, 209, 96, 0.3); }

        .occupied { border-color: rgba(248, 81, 73, 0.4); }
        .occupied .slot-id { color: var(--accent-red); }
        .occupied .slot-status { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid rgba(248, 81, 73, 0.3); }

        .occupied::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle, rgba(248, 81, 73, 0.1) 0%, transparent 70%);
            animation: pulseOccupied 2s infinite;
        }
        @keyframes pulseOccupied {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        /* Controls */
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-shrink: 0;
        }
        button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.8rem;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .btn-save { background: linear-gradient(135deg, #238636, #2ea043); color: white; }
        .btn-save:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(35, 134, 54, 0.4); }
        .btn-reset { background: linear-gradient(135deg, #da3633, #f85149); color: white; }
        .btn-reset:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(218, 54, 51, 0.4); }
        .btn-start { background: linear-gradient(135deg, #1f6feb, #58a6ff); color: white; display: none; }
        .btn-start:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(31, 111, 235, 0.4); }

        /* Status Message */
        .status-msg {
            text-align: center;
            padding: 8px;
            margin-bottom: 10px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.8rem;
            background: rgba(88, 166, 255, 0.1);
            border: 1px solid rgba(88, 166, 255, 0.2);
            color: var(--accent-blue);
            flex-shrink: 0;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .dashboard { flex-direction: column; padding: 10px; }
            .big-availability .value { font-size: 3rem; }
            header { padding: 10px 15px; }
            .logo { font-size: 1rem; }
        }
    </style>
</head>
<body>
    <div class="bg-grid"></div>

    <header>
        <div class="logo">SMART<span>PARK</span></div>
        <div class="system-status" id="systemStatus">
            <div class="live-dot"></div>
            <span id="statusText">Initializing</span>
        </div>
    </header>

    <div class="dashboard">
        <!-- LEFT: Video Feed -->
        <div class="video-section">
            <div class="status-msg" id="statusMsg">
                {% if is_running %}
                    System is RUNNING. Detection Active.
                {% elif is_calibrated %}
                    Calibration Saved. Click 'Start Process' to begin.
                {% else %}
                    Calibration Mode: Click corners for P1-P8.
                {% endif %}
            </div>
            <div class="video-container" id="imageContainer">
                <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Feed">
                <canvas id="overlayCanvas"></canvas>
            </div>
            <div class="controls">
                <button class="btn-save" id="saveBtn" onclick="saveSpots()">Save Calibration</button>
                <button class="btn-start" id="startBtn" onclick="startProcess()">Start Process</button>
                <button class="btn-reset" onclick="resetSpots()">Reset Calibration</button>
            </div>
        </div>

        <!-- RIGHT: Status Panel -->
        <div class="status-section">
            <!-- Big Availability Counter -->
            <div class="big-availability">
                <div class="label">Slots Available</div>
                <div class="value" id="bigAvailable">--</div>
                <div class="sub">Occupied: <span id="bigOccupied">--</span> / Total: <span id="bigTotal">--</span></div>
            </div>

            <!-- Slot Grid -->
            <div class="slot-grid" id="slotGrid">
                <div style="grid-column: span 2; text-align: center; color: var(--text-muted); padding: 20px;">
                    Waiting for calibration...
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('overlayCanvas');
        const ctx = canvas.getContext('2d');
        const img = document.getElementById('videoFeed');
        let points = [];
        let spots = {};
        let labelIndex = 0;

        const labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];

        let isCalibrated = {{ 'true' if is_calibrated else 'false' }};
        let isRunning = {{ 'true' if is_running else 'false' }};

        async function updateStatusPanel() {
            try {
                const response = await fetch('/get_status');
                const data = await response.json();

                // Update Big Availability
                document.getElementById('bigAvailable').innerText = data.stats.available;
                document.getElementById('bigOccupied').innerText = data.stats.occupied;
                document.getElementById('bigTotal').innerText = data.stats.total;

                // Update System Status
                const statusText = document.getElementById('statusText');
                if (data.is_running) {
                    statusText.innerText = "System Active";
                    statusText.style.color = "var(--accent-green)";
                    document.querySelector('.live-dot').style.backgroundColor = "var(--accent-green)";
                } else if (isCalibrated) {
                    statusText.innerText = "Ready to Start";
                    statusText.style.color = "var(--accent-orange)";
                    document.querySelector('.live-dot').style.backgroundColor = "var(--accent-orange)";
                } else {
                    statusText.innerText = "Calibration Mode";
                    statusText.style.color = "var(--accent-blue)";
                    document.querySelector('.live-dot').style.backgroundColor = "var(--accent-blue)";
                }

                // Update Slot Grid
                const grid = document.getElementById('slotGrid');
                if (!data.is_running || Object.keys(data.status).length === 0) {
                    grid.innerHTML = '<div style="grid-column: span 2; text-align: center; color: var(--text-muted); padding: 20px;">System not running.</div>';
                    return;
                }

                let html = '';
                const sortedKeys = Object.keys(data.status).sort();

                sortedKeys.forEach(spot_id => {
                    const status = data.status[spot_id];
                    let statusClass = 'available';
                    let statusText = 'FREE';

                    if (status === 'OCCUPIED') {
                        statusClass = 'occupied';
                        statusText = 'BUSY';
                    }

                    html += `
                        <div class="slot-card ${statusClass}">
                            <div class="slot-id">${spot_id}</div>
                            <div class="slot-status">${statusText}</div>
                        </div>
                    `;
                });
                grid.innerHTML = html;
            } catch (e) {
                console.error("Error fetching status:", e);
            }
        }

        setInterval(updateStatusPanel, 1000);
        if (isRunning) updateStatusPanel();

        // --- UI LOGIC ---
        if (isRunning) {
            document.getElementById('saveBtn').style.display = 'none';
            document.getElementById('startBtn').style.display = 'none';
        } else if (isCalibrated) {
            document.getElementById('saveBtn').style.display = 'none';
            document.getElementById('startBtn').style.display = 'block';
        }

        img.onload = () => {
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            draw();
        };
        window.onresize = () => {
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            draw();
        };

        canvas.addEventListener('click', (e) => {
            if (isCalibrated || isRunning) return; 
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            points.push({x, y});
            draw();

            if (points.length === 4) {
                if (labelIndex < labels.length) {
                    spots[labels[labelIndex]] = points;
                    labelIndex++;
                    points = [];
                }
                if (labelIndex === labels.length) {
                    document.getElementById('statusMsg').innerText = "All 8 spots drawn. Click 'Save Calibration'.";
                }
            }
        });

        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (isCalibrated || isRunning) return; 

            ctx.strokeStyle = '#58a6ff';
            ctx.lineWidth = 2;
            ctx.font = "14px 'Orbitron', sans-serif";
            ctx.fillStyle = "#58a6ff";

            for (const [label, pts] of Object.entries(spots)) {
                ctx.beginPath();
                ctx.moveTo(pts[0].x, pts[0].y);
                ctx.lineTo(pts[1].x, pts[1].y);
                ctx.lineTo(pts[2].x, pts[2].y);
                ctx.lineTo(pts[3].x, pts[3].y);
                ctx.closePath();
                ctx.stroke();
                ctx.fillText(label, pts[0].x + 10, pts[0].y + 20);
            }

            ctx.fillStyle = '#f85149';
            points.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
                ctx.fill();
            });

            if (points.length > 0) {
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }
                ctx.strokeStyle = '#f0883e';
                ctx.stroke();
            }
        }

        async function saveSpots() {
            if (Object.keys(spots).length < 8) {
                alert("Please draw all 8 spots before saving.");
                return;
            }
            const response = await fetch('/save_calibration', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(spots)
            });
            const result = await response.json();
            alert(result.message);
            if (result.status === 'success') location.reload(); 
        }

        async function startProcess() {
            const response = await fetch('/start_process', { method: 'POST' });
            const result = await response.json();
            if (result.status === 'success') {
                isRunning = true;
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('statusMsg').innerText = "System is RUNNING. Detection Active.";
                document.getElementById('statusMsg').style.background = "rgba(35, 209, 96, 0.1)";
                document.getElementById('statusMsg').style.borderColor = "rgba(35, 209, 96, 0.2)";
                document.getElementById('statusMsg').style.color = "var(--accent-green)";
                updateStatusPanel();
            }
        }

        async function resetSpots() {
            const response = await fetch('/reset_calibration', { method: 'POST' });
            const result = await response.json();
            alert(result.message);
            location.reload(); 
        }
    </script>
</body>
</html>
"""

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()

    dashboard_link = "http://localhost:5000"
    print("Sending email notification...")
    send_email_notification(dashboard_link)

    print(f"Starting web server on {dashboard_link}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)