"""
Simple terminal viewer for call logs.
Run: python call_viewer.py
"""
import json
from pathlib import Path
from datetime import datetime

def view_calls():
    logs_dir = Path(__file__).parent / "call_logs"
    
    if not logs_dir.exists():
        print("No call logs found yet. Make some calls first!")
        return
    
    files = sorted(logs_dir.glob("call_*.json"), reverse=True)
    
    if not files:
        print("No call logs found yet. Make some calls first!")
        return
    
    print("\n" + "=" * 60)
    print("  RECENT CALLS")
    print("=" * 60)
    
    for i, filepath in enumerate(files[:10]):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            
            caller = data.get("caller_number", "Unknown")
            duration = data.get("duration_seconds", 0)
            summary = data.get("summary", {})
            
            lead_score = summary.get("lead_score", "?")
            sentiment = summary.get("sentiment", "?")
            caller_name = summary.get("caller_name", "Unknown")
            interest = summary.get("interest", "Not discussed")
            summary_text = summary.get("summary", "No summary available")
            
            print(f"\n  Call #{i+1}")
            print(f"  Phone:     {caller}")
            print(f"  Name:      {caller_name}")
            print(f"  Duration:  {duration}s")
            print(f"  Score:     {lead_score}/10")
            print(f"  Sentiment: {sentiment}")
            print(f"  Interest:  {interest}")
            print(f"  Summary:   {summary_text}")
            print(f"  File:      {filepath.name}")
            print("  " + "-" * 56)
            
        except Exception as e:
            print(f"\n  Error reading {filepath.name}: {e}")
    
    print(f"\n  Total calls: {len(files)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    view_calls()
