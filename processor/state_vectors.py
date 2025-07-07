"""
State vectors processing for Sentinel-1 Level-0 data.

This module handles the extraction and processing of satellite state vectors
from Level-0 packets for orbit determination and SAR processing.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .constants import SPEED_OF_LIGHT, WAVELENGTH

logger = logging.getLogger(__name__)


@dataclass
class StateVector:
    """
    Individual state vector containing position and velocity.
    
    Attributes:
        time: UTC time in seconds since reference epoch
        position: Position vector [x, y, z] in meters (ECEF)
        velocity: Velocity vector [vx, vy, vz] in m/s (ECEF)
    """
    time: float
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    
    def __post_init__(self):
        """Validate state vector data."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        
        if self.position.shape != (3,):
            raise ValueError('Position must be 3-element array')
        if self.velocity.shape != (3,):
            raise ValueError('Velocity must be 3-element array')


class StateVectors:
    """
    Collection of state vectors for satellite orbit processing.
    
    This class manages state vectors extracted from Level-0 packets and provides
    interpolation and orbit calculation functionality.
    """
    
    def __init__(self, packets: Optional[List] = None):
        """
        Initialize StateVectors from packet data.
        
        Args:
            packets: List of L0Packet objects containing state vector data
        """
        self._state_vectors: List[StateVector] = []
        self._is_sorted: bool = True
        
        if packets:
            self._extract_from_packets(packets)
            
    def add_state_vector(self, state_vector: StateVector) -> None:
        """
        Add a state vector to the collection.
        
        Args:
            state_vector: StateVector object to add
        """
        self._state_vectors.append(state_vector)
        self._is_sorted = False
        
    def sort_by_time(self) -> None:
        """Sort state vectors by time."""
        self._state_vectors.sort(key=lambda sv: sv.time)
        self._is_sorted = True
        
    def get_state_vectors(self) -> List[StateVector]:
        """
        Get list of state vectors.
        
        Returns:
            List of StateVector objects
        """
        if not self._is_sorted:
            self.sort_by_time()
        return self._state_vectors.copy()
        
    def get_time_range(self) -> Tuple[float, float]:
        """
        Get the time range covered by state vectors.
        
        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        if not self._state_vectors:
            return 0.0, 0.0
            
        if not self._is_sorted:
            self.sort_by_time()
            
        return self._state_vectors[0].time, self._state_vectors[-1].time
        
    def interpolate_state_vector(self, target_time: float) -> StateVector:
        """
        Interpolate state vector at target time using polynomial interpolation.
        
        Args:
            target_time: Target time for interpolation
            
        Returns:
            Interpolated StateVector object
            
        Raises:
            ValueError: If interpolation is not possible
        """
        if not self._state_vectors:
            raise ValueError('No state vectors available for interpolation')
            
        if not self._is_sorted:
            self.sort_by_time()
            
        # Check if target time is within range
        start_time, end_time = self.get_time_range()
        if target_time < start_time or target_time > end_time:
            logger.warning(f'Target time {target_time} outside range [{start_time}, {end_time}]')
            
        # Find surrounding state vectors
        times = np.array([sv.time for sv in self._state_vectors])
        positions = np.array([sv.position for sv in self._state_vectors])
        velocities = np.array([sv.velocity for sv in self._state_vectors])
        
        # Use polynomial interpolation (typically 6th order for precise orbits)
        interpolated_position = self._polynomial_interpolate(times, positions, target_time)
        interpolated_velocity = self._polynomial_interpolate(times, velocities, target_time)
        
        return StateVector(target_time, interpolated_position, interpolated_velocity)
        
    def _polynomial_interpolate(self, times: np.ndarray, values: np.ndarray,
                              target_time: float, order: int = 6) -> np.ndarray:
        """
        Perform polynomial interpolation.
        
        Args:
            times: Array of time points
            values: Array of values to interpolate (n_points x n_dimensions)
            target_time: Target time for interpolation
            order: Polynomial order
            
        Returns:
            Interpolated values
        """
        n_points = len(times)
        if n_points < order + 1:
            order = n_points - 1
            
        # Find the closest points for interpolation
        idx = np.searchsorted(times, target_time)
        
        # Determine interpolation window
        half_window = order // 2
        start_idx = max(0, idx - half_window)
        end_idx = min(n_points, start_idx + order + 1)
        start_idx = max(0, end_idx - order - 1)
        
        # Extract interpolation data
        interp_times = times[start_idx:end_idx]
        interp_values = values[start_idx:end_idx]
        
        # Perform Lagrange interpolation for each dimension
        result = np.zeros(values.shape[1])
        
        for dim in range(values.shape[1]):
            result[dim] = self._lagrange_interpolate(interp_times, 
                                                   interp_values[:, dim], 
                                                   target_time)
            
        return result
        
    def _lagrange_interpolate(self, x: np.ndarray, y: np.ndarray, 
                            target_x: float) -> float:
        """
        Lagrange polynomial interpolation.
        
        Args:
            x: Array of x values
            y: Array of y values
            target_x: Target x value
            
        Returns:
            Interpolated y value
        """
        n = len(x)
        result = 0.0
        
        for i in range(n):
            # Calculate Lagrange basis polynomial L_i(target_x)
            li = 1.0
            for j in range(n):
                if i != j:
                    li *= (target_x - x[j]) / (x[i] - x[j])
            result += y[i] * li
            
        return result
        
    def get_satellite_velocity_magnitude(self, time: float) -> float:
        """
        Get satellite velocity magnitude at specified time.
        
        Args:
            time: Time for velocity calculation
            
        Returns:
            Velocity magnitude in m/s
        """
        state_vector = self.interpolate_state_vector(time)
        return np.linalg.norm(state_vector.velocity)
        
    def get_satellite_position(self, time: float) -> np.ndarray:
        """
        Get satellite position at specified time.
        
        Args:
            time: Time for position calculation
            
        Returns:
            Position vector [x, y, z] in ECEF coordinates
        """
        state_vector = self.interpolate_state_vector(time)
        return state_vector.position
        
    def get_satellite_velocity(self, time: float) -> np.ndarray:
        """
        Get satellite velocity at specified time.
        
        Args:
            time: Time for velocity calculation
            
        Returns:
            Velocity vector [vx, vy, vz] in ECEF coordinates
        """
        state_vector = self.interpolate_state_vector(time)
        return state_vector.velocity
        
    def calculate_doppler_frequency(self, time: float, target_position: np.ndarray) -> float:
        """
        Calculate Doppler frequency for a target at specified position.
        
        Args:
            time: Time for calculation
            target_position: Target position [x, y, z] in ECEF coordinates
            
        Returns:
            Doppler frequency in Hz
        """
        state_vector = self.interpolate_state_vector(time)
        
        # Vector from satellite to target
        range_vector = target_position - state_vector.position
        range_distance = np.linalg.norm(range_vector)
        
        if range_distance == 0:
            return 0.0
            
        # Unit vector in range direction
        range_unit = range_vector / range_distance
        
        # Doppler frequency = -2 * (v_sat · range_unit) / wavelength
        doppler_freq = -2 * np.dot(state_vector.velocity, range_unit) / WAVELENGTH
        
        return doppler_freq
        
    def calculate_range_rate(self, time: float, target_position: np.ndarray) -> float:
        """
        Calculate range rate to target.
        
        Args:
            time: Time for calculation
            target_position: Target position [x, y, z] in ECEF coordinates
            
        Returns:
            Range rate in m/s (positive = increasing range)
        """
        state_vector = self.interpolate_state_vector(time)
        
        # Vector from satellite to target
        range_vector = target_position - state_vector.position
        range_distance = np.linalg.norm(range_vector)
        
        if range_distance == 0:
            return 0.0
            
        # Unit vector in range direction
        range_unit = range_vector / range_distance
        
        # Range rate = v_sat · range_unit
        range_rate = np.dot(state_vector.velocity, range_unit)
        
        return range_rate
        
    def get_orbital_period(self) -> float:
        """
        Estimate orbital period from state vectors.
        
        Returns:
            Orbital period in seconds
        """
        if len(self._state_vectors) < 2:
            return 0.0
            
        # Use semi-major axis to estimate period
        # For circular orbit: T = 2π√(a³/GM)
        GM_EARTH = 3.986004418e14  # m³/s²
        
        # Calculate average orbital radius
        positions = np.array([sv.position for sv in self._state_vectors])
        radii = np.linalg.norm(positions, axis=1)
        avg_radius = np.mean(radii)
        
        # Estimate period assuming circular orbit
        period = 2 * np.pi * np.sqrt(avg_radius**3 / GM_EARTH)
        
        return period
        
    def print(self) -> None:
        """Print state vectors information."""
        print(f'State Vectors Summary:')
        print(f'  Number of state vectors: {len(self._state_vectors)}')
        
        if self._state_vectors:
            start_time, end_time = self.get_time_range()
            duration = end_time - start_time
            
            print(f'  Time range: {start_time:.3f} to {end_time:.3f} seconds')
            print(f'  Duration: {duration:.3f} seconds')
            
            # Print first few state vectors
            print('  State vectors:')
            for i, sv in enumerate(self._state_vectors[:5]):
                pos_str = f'[{sv.position[0]:.1f}, {sv.position[1]:.1f}, {sv.position[2]:.1f}]'
                vel_str = f'[{sv.velocity[0]:.3f}, {sv.velocity[1]:.3f}, {sv.velocity[2]:.3f}]'
                print(f'    {i+1}: t={sv.time:.3f}s, pos={pos_str}m, vel={vel_str}m/s')
                
            if len(self._state_vectors) > 5:
                print(f'    ... and {len(self._state_vectors) - 5} more')
                
    def _extract_from_packets(self, packets: List) -> None:
        """
        Extract state vectors from L0 packets.
        
        Args:
            packets: List of L0Packet objects
        """
        logger.info(f'Extracting state vectors from {len(packets)} packets')
        
        # State vectors are typically found in auxiliary data packets
        # This is a simplified extraction - real implementation would
        # parse specific packet types containing orbit data
        
        for packet in packets:
            try:
                # Check if this packet contains state vector data
                if self._is_state_vector_packet(packet):
                    state_vector = self._parse_state_vector_packet(packet)
                    if state_vector:
                        self.add_state_vector(state_vector)
                        
            except Exception as e:
                logger.warning(f'Failed to extract state vector from packet {packet.packet_index}: {e}')
                
        logger.info(f'Extracted {len(self._state_vectors)} state vectors')
        
    def _is_state_vector_packet(self, packet) -> bool:
        """
        Check if packet contains state vector data.
        
        Args:
            packet: L0Packet object
            
        Returns:
            True if packet contains state vector data
        """
        # In real implementation, this would check packet type and content
        # For now, use a simple heuristic based on packet properties
        return (hasattr(packet, 'secondary_header') and 
                packet.get_signal_type() in ['ECHO', 'NOISE'])
                
    def _parse_state_vector_packet(self, packet) -> Optional[StateVector]:
        """
        Parse state vector from packet data.
        
        Args:
            packet: L0Packet object
            
        Returns:
            StateVector object or None if parsing fails
        """
        try:
            # Extract time from packet
            packet_time = packet.get_time()
            
            # In real implementation, position and velocity would be extracted
            # from auxiliary data fields in the packet
            # For now, generate synthetic state vectors based on packet time
            
            # Simplified orbital model for demonstration
            # Real implementation would parse actual orbit data
            orbital_radius = 7.0e6  # ~700 km altitude
            orbital_velocity = 7500  # m/s
            
            # Generate position and velocity based on time
            angle = 2 * np.pi * packet_time / 5900  # ~98 min orbit
            
            position = np.array([
                orbital_radius * np.cos(angle),
                orbital_radius * np.sin(angle),
                0.0  # Simplified equatorial orbit
            ])
            
            velocity = np.array([
                -orbital_velocity * np.sin(angle),
                orbital_velocity * np.cos(angle),
                0.0
            ])
            
            return StateVector(packet_time, position, velocity)
            
        except Exception as e:
            logger.error(f'Error parsing state vector from packet: {e}')
            return None
