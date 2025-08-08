#!/usr/bin/env python3
"""
Test the JSON workout functionality
"""

import sys
import os

# Import the run_json_workout function from main
sys.path.append('.')
from main import run_json_workout

if __name__ == "__main__":
    print("🧪 Testing JSON Workout Functionality")
    print("=" * 40)
    
    # Test the function directly
    try:
        run_json_workout()
    except KeyboardInterrupt:
        print("\n👋 Test completed!")
    except Exception as e:
        print(f"❌ Error during test: {e}")
