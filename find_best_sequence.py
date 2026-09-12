import sys
import os
import numpy as np
from pathlib import Path
from src.dashboard.data_streamer import DataStreamerThread

def analyze_sequence(seq):
    print(f"Analyzing sequence {seq}...")
    streamer = DataStreamerThread(source="rellis3d", sequence=seq, rate_hz=100.0)
    streamer._setup_sources()
    
    if not streamer._file_list or streamer.source == "sample":
        print(f"  -> No data for {seq}")
        return None
        
    num_frames = len(streamer._file_list)
    print(f"  -> Found {num_frames} frames")
    
    best_frame = -1
    best_score = -1
    best_info = {}
    
    for i in range(0, num_frames, 20):
        try:
            points, labels = streamer._get_frame_data(i)
            if len(points) == 0:
                continue
                
            maps = streamer._generate_maps(points, [])
            extras = streamer._last_extras
            
            trav = maps["traversability"]
            non_drivable_count = np.sum(trav < 0.40)
            
            centerline_profile = extras.get("centerline_profile", [])
            max_depth = 0
            for bin_data in centerline_profile:
                if bin_data["depth_below_baseline_m"] is not None:
                    max_depth = max(max_depth, bin_data["depth_below_baseline_m"])
            
            # Find the frame with maximum depth
            score = max_depth
                
            if score > best_score:
                best_score = score
                best_frame = i
                best_info = {
                    "non_drivable_cells": int(non_drivable_count),
                    "max_depth_m": float(max_depth)
                }
        except Exception as e:
            pass
            
    if best_frame != -1:
        print(f"  -> Best frame: {best_frame} with {best_info['non_drivable_cells']} cells < 0.40 trav, {best_info['max_depth_m']:.2f}m depth")
        return (seq, best_frame, best_info)
    else:
        print("  -> No good frame found")
        return None

candidates = ["00000", "00002", "00003", "00004"]
results = []
for c in candidates:
    res = analyze_sequence(c)
    if res:
        results.append(res)
        
if results:
    results.sort(key=lambda x: x[2]["max_depth_m"], reverse=True)
    best = results[0]
    print("\n--- WINNER ---")
    print(f"Sequence: {best[0]}, Frame: {best[1]}, Info: {best[2]}")
else:
    print("No winner found.")
