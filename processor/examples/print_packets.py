#!/usr/bin/env python3
"""
Example script demonstrating packet printing functionality.

This script shows how to load and print information about Level-0 packets
from a Sentinel-1 data file.
"""

import sys
import argparse
from pathlib import Path

# Add the processor module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processor import L0Packet


def print_packet_info(filename: str, packet_index: int = 0) -> None:
    """
    Print detailed information about a specific packet.
    
    Args:
        filename: Path to Level-0 data file
        packet_index: Index of packet to examine
    """
    try:
        # Load packets
        print(f'Loading packets from: {filename}')
        packets = L0Packet.get_packets(filename, max_packets=packet_index + 1)
        
        if not packets:
            print('No packets found in file')
            return
            
        if packet_index >= len(packets):
            print(f'Packet index {packet_index} not available (max: {len(packets) - 1})')
            return
            
        # Get specific packet
        packet = packets[packet_index]
        
        print(f'\n=== Packet {packet_index} Information ===')
        
        # Print headers
        packet.print_primary_header()
        print()
        packet.print_secondary_header()
        print()
        packet.print_modes()
        print()
        packet.print_pulse_info()
        
    except Exception as e:
        print(f'Error processing file: {e}')


def print_file_summary(filename: str, max_packets: int = 100) -> None:
    """
    Print summary information about all packets in file.
    
    Args:
        filename: Path to Level-0 data file
        max_packets: Maximum number of packets to analyze
    """
    try:
        print(f'Loading packets from: {filename}')
        packets = L0Packet.get_packets(filename, max_packets=max_packets)
        
        if not packets:
            print('No packets found in file')
            return
            
        print(f'\n=== File Summary ===')
        print(f'Total packets analyzed: {len(packets)}')
        
        # Count by swath
        swath_counts = {}
        signal_type_counts = {}
        polarization_counts = {}
        
        for packet in packets:
            swath = packet.get_swath_name()
            signal_type = packet.get_signal_type()
            polarization = packet.get_polarisation()
            
            swath_counts[swath] = swath_counts.get(swath, 0) + 1
            signal_type_counts[signal_type] = signal_type_counts.get(signal_type, 0) + 1
            polarization_counts[polarization] = polarization_counts.get(polarization, 0) + 1
            
        print('\nSwath distribution:')
        for swath, count in sorted(swath_counts.items()):
            print(f'  {swath}: {count} packets')
            
        print('\nSignal type distribution:')
        for sig_type, count in sorted(signal_type_counts.items()):
            print(f'  {sig_type}: {count} packets')
            
        print('\nPolarization distribution:')
        for pol, count in sorted(polarization_counts.items()):
            print(f'  {pol}: {count} packets')
            
        # Time range
        if packets:
            times = [packet.get_time() for packet in packets]
            print(f'\nTime range: {min(times):.3f} to {max(times):.3f} seconds')
            print(f'Duration: {max(times) - min(times):.3f} seconds')
            
    except Exception as e:
        print(f'Error processing file: {e}')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Print information about Sentinel-1 Level-0 packets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python print_packets.py data.dat --summary
  python print_packets.py data.dat --packet 5
  python print_packets.py data.dat --summary --max-packets 1000
        """
    )
    
    parser.add_argument('filename', help='Path to Level-0 data file')
    parser.add_argument('--packet', type=int, metavar='INDEX',
                       help='Print detailed info for specific packet index')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary of all packets')
    parser.add_argument('--max-packets', type=int, default=100, metavar='N',
                       help='Maximum number of packets to analyze for summary (default: 100)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.filename).exists():
        print(f'Error: File not found: {args.filename}')
        sys.exit(1)
        
    # Execute requested operation
    if args.packet is not None:
        print_packet_info(args.filename, args.packet)
    elif args.summary:
        print_file_summary(args.filename, args.max_packets)
    else:
        # Default: print first packet
        print_packet_info(args.filename, 0)


if __name__ == '__main__':
    main()
