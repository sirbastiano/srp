#!/usr/bin/env python3
"""
Sentinel-1 Parameter Transformations Usage Examples

This script demonstrates practical usage of the parameter transformation library
for decoding Sentinel-1 data packets. It shows real-world examples of how to
extract and interpret parameters from raw bytecode data.
"""

import struct
from typing import Dict, Any
from parameter_transformations import (
    F_REF,
    # All extraction functions
    extract_packet_version_number,
    extract_packet_type,
    extract_secondary_header_flag,
    extract_process_id,
    extract_packet_category,
    extract_sequence_flags,
    extract_packet_sequence_count,
    extract_packet_data_length,
    extract_coarse_time,
    extract_fine_time,
    extract_sync_marker,
    extract_data_take_id,
    extract_ecc_number,
    extract_test_mode,
    extract_rx_channel_id,
    extract_tx_pulse_ramp_rate,
    extract_tx_pulse_start_frequency,
    extract_tx_pulse_length,
    extract_rank,
    extract_pri,
    extract_baq_mode,
    extract_sampling_window_length,
    extract_sampling_window_start_time,
    # Convenience functions
    decode_all_primary_header_parameters,
    decode_all_secondary_header_parameters,
    # User data functions
    extract_bypass_samples,
    ten_bit_unsigned_to_signed_int,
    # Validation functions
    validate_sync_marker,
    validate_packet_version,
    validate_baq_mode,
)


class Sentinel1PacketDecoder:
    """
    A comprehensive decoder for Sentinel-1 data packets.
    
    This class demonstrates how to use the parameter transformation functions
    in a practical application for decoding complete Sentinel-1 packets.
    """
    
    def __init__(self):
        """Initialize the decoder."""
        self.f_ref = F_REF
        
    def decode_packet(self, packet_data: bytes) -> Dict[str, Any]:
        """
        Decode a complete Sentinel-1 packet.
        
        Args:
            packet_data: Raw packet bytes
            
        Returns:
            Dictionary containing all decoded parameters
        """
        result = {}
        
        # Extract primary header (first 6 bytes)
        if len(packet_data) < 6:
            raise ValueError('Packet too short for primary header')
            
        primary_header = packet_data[:6]
        result['primary_header'] = self.decode_primary_header(primary_header)
        
        # Check if there's a secondary header
        has_secondary = result['primary_header']['secondary_header_flag']
        if has_secondary and len(packet_data) >= 68:  # 6 + 62 bytes
            secondary_header = packet_data[6:68]
            result['secondary_header'] = self.decode_secondary_header(secondary_header)
            
            # Extract user data if present
            data_length = result['primary_header']['packet_data_length']
            if len(packet_data) >= 6 + data_length:
                user_data_start = 68  # After primary + secondary headers
                user_data_end = 6 + data_length
                user_data = packet_data[user_data_start:user_data_end]
                result['user_data'] = self.decode_user_data(
                    user_data, 
                    result['secondary_header']
                )
        
        return result
    
    def decode_primary_header(self, header_bytes: bytes) -> Dict[str, Any]:
        """
        Decode primary header parameters.
        
        Args:
            header_bytes: 6-byte primary header
            
        Returns:
            Dictionary of decoded parameters
        """
        params = decode_all_primary_header_parameters(header_bytes)
        
        # Add validation results
        params['is_valid_version'] = validate_packet_version(params['packet_version_number'])
        
        # Add human-readable interpretations
        params['packet_type_str'] = 'TM' if params['packet_type'] == 0 else 'TC'
        params['has_secondary_header'] = bool(params['secondary_header_flag'])
        
        return params
    
    def decode_secondary_header(self, header_bytes: bytes) -> Dict[str, Any]:
        """
        Decode secondary header parameters.
        
        Args:
            header_bytes: 62-byte secondary header
            
        Returns:
            Dictionary of decoded parameters with physical interpretations
        """
        params = decode_all_secondary_header_parameters(header_bytes)
        
        # Add validation results
        params['is_valid_sync'] = validate_sync_marker(params['sync_marker'])
        params['is_valid_baq'] = validate_baq_mode(params['baq_mode'])
        
        # Add derived parameters and interpretations
        params['total_time'] = params['coarse_time'] + params['fine_time']
        
        # Interpret BAQ mode
        baq_mode_names = {
            0: 'Bypass (10-bit)',
            3: 'FDBAQ Mode 3',
            4: 'FDBAQ Mode 4', 
            5: 'FDBAQ Mode 5',
            12: 'FDBAQ Mode 12',
            13: 'FDBAQ Mode 13',
            14: 'FDBAQ Mode 14'
        }
        params['baq_mode_str'] = baq_mode_names.get(params['baq_mode'], 'Unknown')
        
        # Calculate pulse characteristics
        if params['tx_pulse_length_s'] > 0:
            params['pulse_bandwidth'] = abs(params['tx_pulse_ramp_rate_hz_per_s'] * params['tx_pulse_length_s'])
        
        # PRI frequency
        if params['pri_s'] > 0:
            params['pri_frequency_hz'] = 1.0 / params['pri_s']
        
        return params
    
    def decode_user_data(self, user_data: bytes, secondary_header: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode user data section.
        
        Args:
            user_data: Raw user data bytes
            secondary_header: Decoded secondary header for context
            
        Returns:
            Dictionary containing decoded samples and metadata
        """
        result = {
            'raw_size_bytes': len(user_data),
            'samples': []
        }
        
        baq_mode = secondary_header['baq_mode']
        
        if baq_mode == 0:  # Bypass mode - 10-bit samples
            # Calculate number of samples based on data size
            # 5 samples per 8 bytes in bypass mode
            num_samples = (len(user_data) * 8) // 10  # 10 bits per sample
            
            try:
                samples = extract_bypass_samples(user_data, min(num_samples, 100))  # Limit for demo
                result['samples'] = samples
                result['sample_count'] = len(samples)
                result['sample_type'] = '10-bit signed'
                
                if samples:
                    result['sample_stats'] = {
                        'min': min(samples),
                        'max': max(samples),
                        'mean': sum(samples) / len(samples)
                    }
                    
            except Exception as e:
                result['error'] = f'Sample extraction failed: {e}'
                
        else:
            # FDBAQ modes - would need specific decoders
            result['note'] = f'FDBAQ mode {baq_mode} decoding not implemented in this example'
            result['sample_type'] = f'FDBAQ Mode {baq_mode}'
        
        return result


def demonstrate_real_world_usage():
    """Demonstrate real-world usage scenarios."""
    print('Sentinel-1 Parameter Transformations - Real-World Usage Examples')
    print('=' * 70)
    
    decoder = Sentinel1PacketDecoder()
    
    # Create a realistic test packet
    packet = create_realistic_test_packet()
    
    print(f'Decoding packet of {len(packet)} bytes...')
    
    try:
        decoded = decoder.decode_packet(packet)
        
        print('\n1. PRIMARY HEADER ANALYSIS')
        print('-' * 30)
        ph = decoded['primary_header']
        print(f'Packet Type: {ph["packet_type_str"]} (Telemetry)')
        print(f'Process ID: {ph["process_id"]} (Instrument identifier)')
        print(f'Sequence Count: {ph["packet_sequence_count"]}')
        print(f'Data Length: {ph["packet_data_length"]} bytes')
        print(f'Has Secondary Header: {ph["has_secondary_header"]}')
        print(f'Version Valid: {ph["is_valid_version"]}')
        
        if 'secondary_header' in decoded:
            print('\n2. SECONDARY HEADER ANALYSIS')
            print('-' * 32)
            sh = decoded['secondary_header']
            
            print(f'Sync Marker Valid: {sh["is_valid_sync"]}')
            print(f'Data Take ID: {sh["data_take_id"]}')
            print(f'ECC Number: {sh["ecc_number"]}')
            print(f'Test Mode: {sh["test_mode"]}')
            print(f'Rx Channel: {sh["rx_channel_id"]}')
            
            print(f'\nTiming Information:')
            print(f'  Coarse Time: {sh["coarse_time"]} seconds')
            print(f'  Fine Time: {sh["fine_time"]:.6f} seconds')
            print(f'  Total Time: {sh["total_time"]:.6f} seconds')
            
            print(f'\nRadar Parameters:')
            print(f'  TX Pulse Ramp Rate: {sh["tx_pulse_ramp_rate_hz_per_s"]:.3e} Hz/s')
            print(f'  TX Pulse Start Freq: {sh["tx_pulse_start_frequency_hz"]:.3e} Hz')
            print(f'  TX Pulse Length: {sh["tx_pulse_length_s"]:.6e} seconds')
            if 'pulse_bandwidth' in sh:
                print(f'  Calculated Bandwidth: {sh["pulse_bandwidth"]:.3e} Hz')
            
            print(f'\nData Processing:')
            print(f'  BAQ Mode: {sh["baq_mode"]} ({sh["baq_mode_str"]})')
            print(f'  BAQ Mode Valid: {sh["is_valid_baq"]}')
            print(f'  SWL: {sh["sampling_window_length_s"]:.6e} seconds')
            print(f'  SWST: {sh["sampling_window_start_time_s"]:.6e} seconds')
            
            print(f'\nPulse Repetition:')
            print(f'  PRI: {sh["pri_s"]:.6e} seconds')
            if 'pri_frequency_hz' in sh:
                print(f'  PRF: {sh["pri_frequency_hz"]:.3f} Hz')
            print(f'  Rank: {sh["rank"]}')
        
        if 'user_data' in decoded:
            print('\n3. USER DATA ANALYSIS')
            print('-' * 25)
            ud = decoded['user_data']
            print(f'Raw Data Size: {ud["raw_size_bytes"]} bytes')
            print(f'Sample Type: {ud["sample_type"]}')
            
            if 'sample_count' in ud:
                print(f'Sample Count: {ud["sample_count"]}')
                if 'sample_stats' in ud:
                    stats = ud['sample_stats']
                    print(f'Sample Range: {stats["min"]} to {stats["max"]}')
                    print(f'Sample Mean: {stats["mean"]:.2f}')
                    
                # Show first few samples
                if ud['samples']:
                    samples_preview = ud['samples'][:10]
                    print(f'First 10 Samples: {samples_preview}')
        
        print('\n4. VALIDATION SUMMARY')
        print('-' * 23)
        validations = []
        if 'primary_header' in decoded:
            validations.append(f"Primary header version: {'✓' if decoded['primary_header']['is_valid_version'] else '✗'}")
        if 'secondary_header' in decoded:
            sh = decoded['secondary_header']
            validations.append(f"Sync marker: {'✓' if sh['is_valid_sync'] else '✗'}")
            validations.append(f"BAQ mode: {'✓' if sh['is_valid_baq'] else '✗'}")
        
        for validation in validations:
            print(validation)
            
    except Exception as e:
        print(f'❌ Decoding failed: {e}')
        import traceback
        traceback.print_exc()


def create_realistic_test_packet() -> bytes:
    """Create a realistic test packet with proper structure."""
    # Primary header (6 bytes)
    packet_version = 0
    packet_type = 0  # TM
    secondary_header_flag = 1
    process_id = 64  # Typical SAR processor ID
    packet_category = 12  # Science data
    
    sequence_flags = 3  # Standalone packet
    packet_sequence_count = 15432
    
    packet_data_length = 8190  # Will be 8191 actual
    
    # Pack primary header
    word1 = (packet_version << 13) | (packet_type << 12) | \
            (secondary_header_flag << 11) | (process_id << 4) | packet_category
    word2 = (sequence_flags << 14) | packet_sequence_count
    
    primary_header = struct.pack('>HHH', word1, word2, packet_data_length)
    
    # Secondary header (62 bytes)
    secondary_header = bytearray(62)
    
    # Realistic timing
    secondary_header[0:4] = struct.pack('>I', 758947200)  # GPS epoch-based time
    secondary_header[4:6] = struct.pack('>H', 49152)     # ~0.75 second fine time
    
    # Sync marker
    secondary_header[6:10] = struct.pack('>I', 0x352EF853)
    
    # Data take ID
    secondary_header[10:14] = struct.pack('>I', 123456)
    
    # ECC and other fields
    secondary_header[14] = 1  # ECC
    secondary_header[15] = (0 << 4) | 1  # Test mode 0, Rx channel 1
    
    # TXPRR - realistic chirp rate (positive)
    secondary_header[36:38] = struct.pack('>H', 0x5000)  # Positive, moderate magnitude
    
    # TXPSF - realistic start frequency
    secondary_header[38:40] = struct.pack('>H', 0x6000)  # Positive frequency offset
    
    # Pulse length - realistic value
    pulse_length_raw = int(50e-6 * F_REF)  # 50 microsecond pulse
    secondary_header[40:43] = struct.pack('>I', pulse_length_raw)[1:]  # 24-bit value
    
    # Rank
    secondary_header[43] = (5 << 3) | 0  # Rank 5
    
    # PRI - realistic value
    pri_raw = int(1.0/1200 * F_REF)  # 1200 Hz PRF
    secondary_header[52:56] = struct.pack('>I', pri_raw)
    
    # SWST - realistic receive window start
    swst_raw = int(100e-6 * F_REF)  # 100 microseconds
    secondary_header[56:59] = struct.pack('>I', swst_raw)[1:]  # 24-bit
    
    # SWL - realistic receive window length  
    secondary_header[59:61] = struct.pack('>H', 1500)
    
    # BAQ mode - bypass for simplicity
    secondary_header[61] = 0  # BAQ mode 0 (bypass)
    
    # User data - create some realistic samples
    user_data = create_realistic_sample_data(8191 - 62)  # Remaining space
    
    return primary_header + bytes(secondary_header) + user_data


def create_realistic_sample_data(size_bytes: int) -> bytes:
    """Create realistic sample data for testing."""
    # For bypass mode, create 10-bit samples packed into bytes
    # This creates a simple pattern that mimics radar return data
    
    import math
    data = bytearray()
    
    # Generate samples with some realistic characteristics
    for i in range(size_bytes // 2):  # Rough approximation
        # Create a pattern that mimics radar returns
        amplitude = int(200 * math.sin(i * 0.1) + 100 * math.cos(i * 0.05))
        amplitude = max(-512, min(511, amplitude))  # Clamp to 10-bit signed range
        
        # Convert to 10-bit unsigned
        if amplitude < 0:
            amplitude = 1024 + amplitude
        
        # Pack into bytes (simplified packing)
        data.extend(struct.pack('>H', amplitude & 0x3FF))
    
    return bytes(data[:size_bytes])


if __name__ == '__main__':
    demonstrate_real_world_usage()
