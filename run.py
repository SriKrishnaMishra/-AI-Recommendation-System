#!/usr/bin/env python3
"""
Run script for the AI Recommendation System
"""

import sys
import os

# Add the recommendation_system directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'recommendation_system'))

from app import app

if __name__ == '__main__':
    print("🚀 Starting AI Recommendation System...")
    print("📊 Finance, Health, and Learning AI modules loaded")
    print("🌐 Server will be available at: http://localhost:5000")
    print("🔧 Debug mode: ON")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
