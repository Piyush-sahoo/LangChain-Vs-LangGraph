"""
Research Assistant Project Runner
This script helps you run both LangChain and LangGraph applications easily.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_langchain_app():
    """Run the LangChain Researcher app"""
    print("🚀 Starting LangChain Researcher...")
    print("📍 URL: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    os.chdir("langchain_researcher")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])

def run_langgraph_app():
    """Run the LangGraph Researcher app"""
    print("🧠 Starting LangGraph Self-Correcting Researcher...")
    print("📍 URL: http://localhost:8502")
    print("Press Ctrl+C to stop\n")
    
    os.chdir("langgraph_researcher")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502"])

def main():
    print("=" * 60)
    print("🔬 Research Assistant Project")
    print("=" * 60)
    print()
    print("Choose which application to run:")
    print()
    print("1. 🚀 LangChain Researcher (Port 8501)")
    print("   - Fast, predictable research")
    print("   - Linear workflow demonstration")
    print()
    print("2. 🧠 LangGraph Self-Correcting Researcher (Port 8502)")
    print("   - Intelligent, self-improving research")
    print("   - Cyclical workflow with evaluation")
    print()
    print("3. 📚 View Project Documentation")
    print()
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-3): ").strip()
            
            if choice == "1":
                run_langchain_app()
                break
            elif choice == "2":
                run_langgraph_app()
                break
            elif choice == "3":
                print("\n📚 Project Documentation:")
                print("-" * 40)
                print("This project demonstrates the key differences between:")
                print()
                print("🚀 LangChain:")
                print("  • Linear, predictable workflows")
                print("  • Fast execution")
                print("  • Assembly-line approach")
                print("  • Cannot self-correct")
                print()
                print("🧠 LangGraph:")
                print("  • Cyclical, intelligent workflows")
                print("  • Self-evaluation and improvement")
                print("  • Adaptive decision making")
                print("  • Higher quality output")
                print()
                print("Both apps use:")
                print("  • Gemini 2.0 Flash for LLM")
                print("  • Tavily for web search")
                print("  • Streamlit for UI")
                print()
                input("Press Enter to continue...")
                print()
            elif choice == "0":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 0, 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()