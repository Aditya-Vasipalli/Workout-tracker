#!/usr/bin/env python3
"""
Main Launcher for AI Workout Tracker
Choose between different versions and modes
"""

import sys
import os

def show_menu():
    """Display main menu"""
    print("AI WORKOUT TRACKER")
    print("=" * 40)
    print()
    print("Choose your workout mode:")
    print("1. 🚀 Enhanced Tracker (25+ exercises)")
    print("2. 🤖 MoveNet Tracker (55+ exercises)")
    print("3. � MoveNet with STRICT Form Validation")
    print("4. �📋 Run JSON Workout")
    print("5. 📖 Read workout log")
    print("6. 📚 View supported exercises")
    print("7. ❌ Exit")
    print()

def run_json_workout():
    """Run a workout from JSON file"""
    try:
        from movenet_tracker import MoveNetWorkoutTracker
        tracker = MoveNetWorkoutTracker()
        
        # Show available JSON files
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if json_files:
            print("\n📋 Available workout files:")
            for i, filename in enumerate(json_files, 1):
                print(f"  {i}. {filename}")
            print(f"  {len(json_files) + 1}. Enter custom filename")
        
        choice = input(f"\nSelect workout file (1-{len(json_files) + 1}) or enter filename: ").strip()
        
        # Handle selection
        if choice.isdigit() and 1 <= int(choice) <= len(json_files):
            filename = json_files[int(choice) - 1]
        elif choice.isdigit() and int(choice) == len(json_files) + 1:
            filename = input("Enter JSON filename: ").strip()
        else:
            filename = choice
        
        # Load workout data first to show camera guidance
        workout_data = tracker.load_workout_from_json(filename)
        if workout_data:
            # Ask about camera setup
            setup_camera = input("\n📹 Do you want camera positioning help? (y/n): ").strip().lower()
            if setup_camera == 'y':
                print("\n🎯 Camera setup options:")
                print("1. 📹 Select different camera")
                print("2. 🎯 Test camera view")
                print("3. ▶️  Start workout now")
                
                camera_choice = input("Choose (1-3): ").strip()
                if camera_choice == '1':
                    tracker.select_camera()
                elif camera_choice == '2':
                    tracker.test_camera_view()
            
            # Run the workout
            tracker.run_workout_program(workout_data)
        
    except ImportError as e:
        print(f"❌ Error importing MoveNet tracker: {e}")
        print("Make sure TensorFlow is installed.")
    except Exception as e:
        print(f"❌ Error running JSON workout: {e}")

def run_movenet_tracker():
    """Run the MoveNet workout tracker"""
    try:
        from movenet_tracker import main
        main()
    except ImportError as e:
        print(f"❌ Error importing MoveNet tracker: {e}")
        print("Make sure TensorFlow is installed.")
    except Exception as e:
        print(f"❌ Error running MoveNet tracker: {e}")

def run_strict_movenet_tracker():
    """Run the MoveNet tracker with strict progressive form validation"""
    try:
        from movenet_tracker import MoveNetWorkoutTracker
        
        print("🔒 STRICT FORM VALIDATION MODE")
        print("=" * 40)
        print("This mode uses progressive movement tracking to ensure:")
        print("• Proper form (180°→150°→90°→60°→90°→150°→180°)")
        print("• Controlled movement timing")
        print("• No false reps from random movement")
        print("• Injury prevention through strict validation")
        print()
        
        confirm = input("Enable strict form validation? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Returning to regular MoveNet tracker...")
            run_movenet_tracker()
            return
        
        # Initialize tracker with strict form validation
        tracker = MoveNetWorkoutTracker()
        tracker.enable_strict_form_validation(True)
        
        # Run the main tracking interface
        from movenet_tracker import main
        main()
        
    except ImportError as e:
        print(f"❌ Error importing MoveNet tracker: {e}")
        print("Make sure TensorFlow is installed.")
    except Exception as e:
        print(f"❌ Error running strict MoveNet tracker: {e}")

def run_enhanced_tracker():
    """Run the enhanced workout tracker"""
    try:
        from enhanced_workout_tracker import main
        main()
    except ImportError as e:
        print(f"❌ Error importing enhanced tracker: {e}")
        print("Make sure MediaPipe is installed.")

def read_workout_log():
    """Read and display workout log"""
    try:
        from enhanced_workout_tracker import read_log
        filename = input("Enter log filename (or press Enter for most recent): ").strip()
        if filename:
            read_log(filename)
        else:
            read_log()
    except ImportError:
        print("❌ Enhanced tracker not available")

def show_supported_exercises():
    """Show all supported exercises"""
    try:
        from enhanced_workout_tracker import EnhancedWorkoutTracker
        tracker = EnhancedWorkoutTracker()
        tracker.show_supported_exercises()
    except ImportError:
        print("❌ Enhanced tracker not available")

def main():
    """Main program"""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                run_enhanced_tracker()
            elif choice == '2':
                run_movenet_tracker()
            elif choice == '3':
                run_strict_movenet_tracker()
            elif choice == '4':
                run_json_workout()
            elif choice == '5':
                read_workout_log()
            elif choice == '6':
                show_supported_exercises()
            elif choice == '7':
                print("👋 Goodbye! Stay fit!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")
            
            input("\nPress Enter to continue...")
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Stay fit!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
