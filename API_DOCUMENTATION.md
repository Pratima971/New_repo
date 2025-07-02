# Accident Detection System - API Documentation

## 🚀 **System Access Methods**

### 1. **Web Interface Access**
- **URL**: `http://YOUR_IP_ADDRESS:5000`
- **Description**: Full web interface with upload, analysis, and visualization
- **Usage**: Direct browser access for manual testing and analysis

### 2. **API Access for External Systems**
Base URL: `http://YOUR_IP_ADDRESS:5000/api`

## 📡 **API Endpoints**

### **Health Check**
```
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "Accident Detection System",
  "model_loaded": true,
  "timestamp": 1751043399.123
}
```

### **System Information**
```
GET /api/info
```
**Response:**
```json
{
  "system": "Accident Detection System",
  "version": "1.0.0",
  "supported_formats": {
    "images": [".jpg", ".jpeg", ".png", ".bmp"],
    "videos": [".mp4", ".avi", ".mov", ".mkv"]
  },
  "endpoints": {
    "web_interface": "/",
    "health_check": "/api/health",
    "analyze_file": "/api/analyze",
    "system_info": "/api/info"
  }
}
```

### **Analyze File**
```
POST /api/analyze
Content-Type: multipart/form-data
```
**Parameters:**
- `file`: Image or video file to analyze

**Response for Images:**
```json
{
  "accident_detected": true,
  "message": "Accident detected!",
  "detections": [
    {
      "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
      "confidence": 0.85,
      "impact": "Severe",
      "class_id": 0
    }
  ],
  "total_detections": 1,
  "image_dimensions": {"width": 640, "height": 480},
  "annotated_image": "data:image/jpeg;base64,..."
}
```

**Response for Videos:**
```json
{
  "accident_detected": true,
  "message": "Accident detected!",
  "detections": [...],
  "total_detections": 5,
  "frames_processed": 150,
  "impact_summary": {
    "overall_impact": "Severe",
    "severe_count": 2,
    "moderate_count": 2,
    "minor_count": 1
  },
  "annotated_video": "/annotated_video_150_1751043399.mp4",
  "key_frames": [...]
}
```

## 🔧 **Integration Examples**

### **Python Integration**
```python
import requests

# Health check
response = requests.get('http://YOUR_IP:5000/api/health')
print(response.json())

# Analyze image
with open('accident_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://YOUR_IP:5000/api/analyze', files=files)
    result = response.json()
    print(f"Accident detected: {result['accident_detected']}")
```

### **cURL Integration**
```bash
# Health check
curl http://YOUR_IP:5000/api/health

# Analyze file
curl -X POST -F "file=@accident_video.mp4" http://YOUR_IP:5000/api/analyze
```

### **JavaScript Integration**
```javascript
// Health check
fetch('http://YOUR_IP:5000/api/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Analyze file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://YOUR_IP:5000/api/analyze', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🌐 **Network Configuration**

### **Find Your IP Address**
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

### **Access URLs**
- **Local**: `http://127.0.0.1:5000`
- **Network**: `http://YOUR_LOCAL_IP:5000`
- **Example**: `http://192.168.1.100:5000`

### **Firewall Configuration**
Make sure port 5000 is open in your firewall for external access.

## 🔒 **Security Considerations**

1. **Local Network Only**: Currently configured for local network access
2. **No Authentication**: Add authentication for production use
3. **File Size Limits**: Consider adding file size restrictions
4. **Rate Limiting**: Implement rate limiting for production

## 📊 **Response Codes**

- `200`: Success
- `400`: Bad request (missing file, invalid format)
- `404`: Endpoint not found
- `500`: Server error

## 🎯 **Use Cases**

1. **Security Systems**: Integrate with CCTV monitoring
2. **Traffic Management**: Real-time accident detection
3. **Insurance**: Automated claim processing
4. **Fleet Management**: Vehicle safety monitoring
5. **Research**: Accident analysis and statistics
