from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import tempfile
import time

app = Flask(__name__, static_folder='.', static_url_path='')

# Load the YOLO model
try:
    model = YOLO('best (1).pt')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.before_request
def check_model():
    if model is None and request.endpoint not in ['index', 'static']:
        return jsonify({'error': 'Model not loaded properly'}), 500

@app.route('/')
def index():
    return app.send_static_file('front.html')

@app.route('/annotated_video_<path:filename>')
def serve_video(filename):
    """Serve annotated video files"""
    try:
        return app.send_static_file(f'annotated_video_{filename}')
    except Exception as e:
        return jsonify({'error': f'Video not found: {str(e)}'}), 404

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        file.save(temp_file.name)
        temp_file.close()

        # Open video for processing
        cap = cv2.VideoCapture(temp_file.name)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video with annotations
        output_path = temp_file.name.replace('.mp4', '_annotated.mp4')
        # Try different codecs for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # If XVID fails, try mp4v
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Analyze frames with skipping for faster processing
        all_detections = []
        accident_detected = False
        confidence_threshold = 0.7  # Increased threshold to reduce false positives
        frame_count = 0
        frame_skip = 5  # Process every 5th frame for speed

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for faster processing
            if frame_count % frame_skip != 0:
                out.write(frame)  # Write original frame without processing
                continue

            # Process frame with YOLO model
            results = model(frame)
            frame_detections = []

            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf > confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Additional filtering: check bounding box size (avoid tiny detections)
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        frame_area = width * height

                        # Only consider detections that are at least 1% of frame area
                        if box_area < (frame_area * 0.01):
                            continue

                        accident_detected = True

                        # Determine impact level based on confidence (stricter thresholds)
                        if conf > 0.85:
                            impact = "Severe"
                            color = (0, 0, 255)  # Red
                        elif conf > 0.75:
                            impact = "Moderate"
                            color = (0, 165, 255)  # Orange
                        else:
                            impact = "Minor"
                            color = (0, 255, 255)  # Yellow

                        # Draw bounding box on frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                        # Add impact label with confidence points
                        confidence_points = f"{conf:.2f}"  # Show confidence as decimal points
                        label = f"{impact} Impact: {confidence_points}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                     (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        frame_detections.append({
                            'bbox': {
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            },
                            'confidence': round(conf, 3),
                            'impact': impact,
                            'frame': frame_count
                        })

            # Write annotated frame to output video
            out.write(frame)

            if frame_detections:
                all_detections.extend(frame_detections)

        # Release video objects
        cap.release()
        out.release()

        # Store annotated video path for serving and extract key frames
        annotated_video_url = None
        key_frames = []

        if accident_detected:
            # Move annotated video to static folder for serving
            import shutil
            static_video_path = f"annotated_video_{frame_count}_{int(time.time())}.mp4"
            shutil.move(output_path, static_video_path)
            annotated_video_url = f"/{static_video_path}"

            # Extract key frames with accidents for display
            cap_extract = cv2.VideoCapture(static_video_path)
            frame_idx = 0
            while True:
                ret, frame = cap_extract.read()
                if not ret:
                    break
                frame_idx += 1

                # Check if this frame has accidents
                frame_has_accident = any(d['frame'] == frame_idx for d in all_detections)
                if frame_has_accident and len(key_frames) < 5:  # Limit to 5 key frames
                    # Convert frame to base64
                    import base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    key_frames.append({
                        'frame_number': frame_idx,
                        'image_data': f"data:image/jpeg;base64,{frame_b64}",
                        'detections': [d for d in all_detections if d['frame'] == frame_idx]
                    })
            cap_extract.release()

        elif os.path.exists(output_path):
            os.unlink(output_path)

        # Clean up the temporary files
        os.unlink(temp_file.name)

        # Calculate summary statistics
        total_detections = len(all_detections)
        severe_count = len([d for d in all_detections if d['impact'] == 'Severe'])
        moderate_count = len([d for d in all_detections if d['impact'] == 'Moderate'])
        minor_count = len([d for d in all_detections if d['impact'] == 'Minor'])

        # Determine overall impact
        if severe_count > 0:
            overall_impact = "Severe"
        elif moderate_count > 0:
            overall_impact = "Moderate"
        elif minor_count > 0:
            overall_impact = "Minor"
        else:
            overall_impact = "None"

        response_data = {
            'accident_detected': accident_detected,
            'message': 'Accident detected!' if accident_detected else 'No accident detected',
            'detections': all_detections,
            'total_detections': total_detections,
            'frames_processed': frame_count,
            'impact_summary': {
                'overall_impact': overall_impact,
                'severe_count': severe_count,
                'moderate_count': moderate_count,
                'minor_count': minor_count
            }
        }

        if annotated_video_url:
            response_data['annotated_video'] = annotated_video_url

        if key_frames:
            response_data['key_frames'] = key_frames

        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/webcam_feed')
def webcam_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame with YOLO model
            results = model(frame)
            
            # Check if accident detected
            accident_detected = False
            confidence_threshold = 0.7  # Increased threshold to reduce false positives
            
            if len(results[0].boxes) > 0 and any(conf > confidence_threshold for conf in results[0].boxes.conf):
                accident_detected = True
                # Draw red border if accident detected
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Send frame and detection status
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'X-Accident: ' + str(accident_detected).encode() + b'\r\n\r\n' + 
                   frame_bytes + b'\r\n')
    
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_webcam_accident')
def check_webcam_accident():
    # This endpoint will be polled by frontend to check accident status
    # In a real implementation, you would need a way to share state between routes
    # For now, this is a placeholder
    return jsonify({'accident_detected': False})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()

        # Save uploaded file to a temporary location with proper extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        file.save(temp_file.name)
        temp_file.close()

        # Read the image to get dimensions
        img = cv2.imread(temp_file.name)
        img_height, img_width = img.shape[:2]

        # Process the image with YOLO model with smaller batch size
        results = model(temp_file.name, batch=1)

        # Extract detailed detection information
        detections = []
        accident_detected = False
        confidence_threshold = 0.7  # Increased threshold to reduce false positives

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf > confidence_threshold:
                    # Get bounding box coordinates (normalized to 0-1)
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # Convert to pixel coordinates
                    x1_px = int(x1)
                    y1_px = int(y1)
                    x2_px = int(x2)
                    y2_px = int(y2)

                    # Additional filtering: check bounding box size (avoid tiny detections)
                    box_width = x2_px - x1_px
                    box_height = y2_px - y1_px
                    box_area = box_width * box_height
                    image_area = img.shape[0] * img.shape[1]

                    # Only consider detections that are at least 1% of image area
                    if box_area < (image_area * 0.01):
                        continue

                    accident_detected = True

                    # Calculate impact level based on confidence and box size
                    box_area = (x2_px - x1_px) * (y2_px - y1_px)
                    total_area = img_width * img_height
                    area_ratio = box_area / total_area

                    # Determine impact level (stricter thresholds)
                    if conf > 0.85 and area_ratio > 0.3:
                        impact = "Severe"
                    elif conf > 0.75 and area_ratio > 0.2:
                        impact = "Moderate"
                    else:
                        impact = "Minor"

                    detections.append({
                        'bbox': {
                            'x1': x1_px,
                            'y1': y1_px,
                            'x2': x2_px,
                            'y2': y2_px
                        },
                        'confidence': round(conf, 3),
                        'impact': impact,
                        'class_id': int(boxes.cls[i]) if hasattr(boxes, 'cls') else 0
                    })

        # Create annotated image with bounding boxes
        annotated_img_path = None
        if accident_detected:
            annotated_img = img.copy()
            for detection in detections:
                bbox = detection['bbox']
                impact = detection['impact']

                # Choose color based on impact
                if impact == "Severe":
                    color = (0, 0, 255)  # Red
                elif impact == "Moderate":
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow

                # Draw bounding box
                cv2.rectangle(annotated_img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 3)

                # Add impact label with confidence points
                confidence_points = f"{conf:.2f}"  # Show confidence as decimal points
                label = f"{impact} Impact: {confidence_points}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(annotated_img, (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                             (bbox['x1'] + label_size[0], bbox['y1']), color, -1)
                cv2.putText(annotated_img, label, (bbox['x1'], bbox['y1'] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save annotated image
            annotated_img_path = temp_file.name.replace(file_ext, '_annotated.jpg')
            cv2.imwrite(annotated_img_path, annotated_img)

        # Clean up the original temporary file
        os.unlink(temp_file.name)

        response_data = {
            'accident_detected': accident_detected,
            'message': 'Accident detected!' if accident_detected else 'No accident detected',
            'detections': detections,
            'total_detections': len(detections),
            'image_dimensions': {'width': img_width, 'height': img_height}
        }

        if annotated_img_path:
            # Convert annotated image to base64 for frontend display
            import base64
            with open(annotated_img_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                response_data['annotated_image'] = f"data:image/jpeg;base64,{img_data}"
            os.unlink(annotated_img_path)

        return jsonify(response_data)

    except Exception as e:
        # Log the error for debugging
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/model_status')
def model_status():
    try:
        # Try to run a simple inference to check if model works
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = model(dummy_img)
        return jsonify({
            'status': 'ok',
            'model_loaded': True,
            'model_path': model.ckpt_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model_loaded': False
        })

@app.route('/test_image')
def test_image():
    try:
        # Create a simple test image
        test_img_path = os.path.join(tempfile.gettempdir(), 'test_image.jpg')
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[100:200, 100:200, :] = 255  # White square in the middle
        cv2.imwrite(test_img_path, img)

        # Process with model
        results = model(test_img_path)

        # Clean up
        os.unlink(test_img_path)

        return jsonify({
            'status': 'success',
            'message': 'Test image processed successfully',
            'detections': len(results[0].boxes)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing test image: {str(e)}'
        })

# API endpoints for external system integration
@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint for external systems"""
    return jsonify({
        'status': 'healthy',
        'service': 'Accident Detection System',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for external systems to analyze images/videos"""
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Determine file type
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Process as image
            return upload_image()
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Process as video
            return upload_video()
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

    except Exception as e:
        return jsonify({'error': f'API error: {str(e)}'}), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """System information endpoint"""
    return jsonify({
        'system': 'Accident Detection System',
        'version': '1.0.0',
        'supported_formats': {
            'images': ['.jpg', '.jpeg', '.png', '.bmp'],
            'videos': ['.mp4', '.avi', '.mov', '.mkv']
        },
        'endpoints': {
            'web_interface': '/',
            'health_check': '/api/health',
            'analyze_file': '/api/analyze',
            'system_info': '/api/info'
        },
        'features': [
            'Real-time accident detection',
            'Bounding box visualization',
            'Impact severity assessment',
            'Video frame analysis',
            'Key frame extraction'
        ]
    })

if __name__ == '__main__':
    # Make accessible to other systems on the network
    app.run(host='0.0.0.0', port=5000, debug=True)







