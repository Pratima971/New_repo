#!/usr/bin/env python3
"""
Example client for accessing the Accident Detection System API
This demonstrates how other systems can integrate with your accident detection service.
"""

import requests
import json
import time

# Configuration
SERVER_IP = "192.168.1.6"  # Your server's IP address
SERVER_PORT = 5000
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

def check_health():
    """Check if the accident detection service is healthy"""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Service Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   Service: {data['service']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to service: {e}")
        return False

def get_system_info():
    """Get system information and capabilities"""
    try:
        response = requests.get(f"{BASE_URL}/api/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\n📋 System Information:")
            print(f"   System: {data['system']}")
            print(f"   Version: {data['version']}")
            print(f"   Supported Image Formats: {', '.join(data['supported_formats']['images'])}")
            print(f"   Supported Video Formats: {', '.join(data['supported_formats']['videos'])}")
            print("   Available Endpoints:")
            for name, endpoint in data['endpoints'].items():
                print(f"     - {name}: {endpoint}")
            return True
        else:
            print(f"❌ Failed to get system info: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot get system info: {e}")
        return False

def analyze_file(file_path):
    """Analyze an image or video file for accidents"""
    try:
        print(f"\n🔍 Analyzing file: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("📊 Analysis Results:")
            print(f"   Accident Detected: {'🚨 YES' if data['accident_detected'] else '✅ NO'}")
            print(f"   Message: {data['message']}")
            print(f"   Total Detections: {data['total_detections']}")
            
            if 'frames_processed' in data:
                print(f"   Frames Processed: {data['frames_processed']}")
                
            if 'impact_summary' in data:
                impact = data['impact_summary']
                print(f"   Overall Impact: {impact['overall_impact']}")
                if impact['severe_count'] > 0:
                    print(f"   Severe Accidents: {impact['severe_count']}")
                if impact['moderate_count'] > 0:
                    print(f"   Moderate Accidents: {impact['moderate_count']}")
                if impact['minor_count'] > 0:
                    print(f"   Minor Accidents: {impact['minor_count']}")
            
            if data['detections']:
                print("   Individual Detections:")
                for i, detection in enumerate(data['detections'][:3]):  # Show first 3
                    print(f"     Detection {i+1}:")
                    print(f"       Impact: {detection['impact']}")
                    print(f"       Confidence: {detection['confidence']:.3f}")
                    if 'frame' in detection:
                        print(f"       Frame: {detection['frame']}")
                
                if len(data['detections']) > 3:
                    print(f"     ... and {len(data['detections']) - 3} more detections")
            
            return data
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    """Main function demonstrating API usage"""
    print("🚗 Accident Detection System - Client Example")
    print("=" * 50)
    
    # Check if service is available
    if not check_health():
        print("\n❌ Service is not available. Please ensure:")
        print("   1. The accident detection server is running")
        print("   2. The server IP address is correct")
        print("   3. Port 5000 is accessible")
        return
    
    # Get system information
    get_system_info()
    
    # Example file analysis (you can replace with actual file paths)
    print("\n" + "=" * 50)
    print("📁 File Analysis Examples:")
    print("   (Replace these paths with actual image/video files)")
    
    # Example paths - replace with actual files
    example_files = [
        "accident_image.jpg",
        "traffic_video.mp4",
        "normal_traffic.jpg"
    ]
    
    for file_path in example_files:
        print(f"\n📄 Example: {file_path}")
        print("   (File not found - this is just an example)")
        # Uncomment the line below to analyze actual files:
        # analyze_file(file_path)
    
    print("\n" + "=" * 50)
    print("🌐 Access Information:")
    print(f"   Web Interface: http://{SERVER_IP}:{SERVER_PORT}")
    print(f"   API Base URL: {BASE_URL}/api")
    print(f"   Health Check: {BASE_URL}/api/health")
    print(f"   System Info: {BASE_URL}/api/info")
    print(f"   Analyze File: {BASE_URL}/api/analyze (POST)")

if __name__ == "__main__":
    main()
