#!/usr/bin/env python3
"""
Debug script to test decoder functionality with real data.

This script helps debug issues with packet decoding and processing.
"""

import sys
import numpy as np
from pathlib import Path

# Add the processor module to path
sys.path.insert(0, str(Path(__file__).parent))

from processor import S1Decoder, L0Packet


def debug_packets(filename: str, max_packets: int = 10):
    """
    Debug packet loading and decoding.
    
    Args:
        filename: Path to data file
        max_packets: Maximum packets to analyze
    """
    print(f"=== Debugging packets from: {filename} ===")
    
    try:
        # Load packets
        print(f"Loading packets (max: {max_packets})...")
        packets = L0Packet.get_packets(filename, max_packets=max_packets)
        print(f"Loaded {len(packets)} packets")
        
        if not packets:
            print("No packets found!")
            return
            
        # Analyze first few packets
        print("\n=== Packet Analysis ===")
        for i, packet in enumerate(packets[:5]):
            print(f"\nPacket {i}:")
            print(f"  Swath: {packet.get_swath_name()}")
            print(f"  Signal Type: {packet.get_signal_type()}")
            print(f"  BAQ Mode: {packet.get_baq_mode()}")
            print(f"  Data Length: {len(packet._raw_data) if packet._raw_data else 0} bytes")
            
            # Try to decode
            try:
                decoded = packet.decode_packet_data()
                print(f"  Decoded Shape: {decoded.shape}")
                print(f"  Decoded Type: {decoded.dtype}")
                if decoded.size > 0:
                    print(f"  Data Range: {np.min(np.abs(decoded)):.2e} to {np.max(np.abs(decoded)):.2e}")
                else:
                    print("  Empty decoded data")
            except Exception as e:
                print(f"  Decode Error: {e}")
                
        # Count by swath and signal type
        print("\n=== Packet Distribution ===")
        swath_counts = {}
        signal_counts = {}
        baq_counts = {}
        
        for packet in packets:
            swath = packet.get_swath_name()
            signal = packet.get_signal_type()
            baq = packet.get_baq_mode()
            
            swath_counts[swath] = swath_counts.get(swath, 0) + 1
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            baq_counts[baq] = baq_counts.get(baq, 0) + 1
            
        print("Swaths:", dict(sorted(swath_counts.items())))
        print("Signal Types:", dict(sorted(signal_counts.items())))
        print("BAQ Modes:", dict(sorted(baq_counts.items())))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def debug_decoder(filename: str):
    """
    Debug decoder functionality.
    
    Args:
        filename: Path to data file
    """
    print(f"\n=== Debugging decoder with: {filename} ===")
    
    try:
        # Initialize decoder
        print("Initializing decoder...")
        decoder = S1Decoder(filename)
        
        # Get available swaths
        swaths = decoder.get_swath_names()
        print(f"Available swaths: {swaths}")
        
        if not swaths:
            print("No swaths found!")
            return
            
        # Try processing first swath
        first_swath = swaths[0]
        print(f"\nTrying to process swath: {first_swath}")
        
        try:
            # Get raw data
            print("Getting raw swath data...")
            raw_data = decoder.get_swath(first_swath)
            print(f"Raw data shape: {raw_data.shape}")
            print(f"Raw data type: {raw_data.dtype}")
            
            if raw_data.size > 0:
                print(f"Data range: {np.min(np.abs(raw_data)):.2e} to {np.max(np.abs(raw_data)):.2e}")
                
                # Try range compression
                print("Trying range compression...")
                range_compressed = decoder.get_range_compressed_swath(first_swath)
                print(f"Range compressed shape: {range_compressed.shape}")
                
                # Try azimuth compression
                print("Trying azimuth compression...")
                azimuth_compressed = decoder.get_azimuth_compressed_swath(first_swath)
                print(f"Azimuth compressed shape: {azimuth_compressed.shape}")
                
            else:
                print("No data in swath!")
                
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Decoder error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python debug_processor.py <data_file> [max_packets]")
        sys.exit(1)
        
    filename = sys.argv[1]
    max_packets = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not Path(filename).exists():
        print(f"File not found: {filename}")
        sys.exit(1)
        
    # Debug packets
    debug_packets(filename, max_packets)
    
    # Debug decoder
    debug_decoder(filename)


if __name__ == '__main__':
    main()
