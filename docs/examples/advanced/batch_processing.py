#!/usr/bin/env python3
"""
SARPYX Batch Processing Example
==============================

This example demonstrates comprehensive batch processing capabilities for large-scale
SAR data processing using SARPYX. It covers distributed processing, resource management,
job scheduling, monitoring, and optimization strategies.

Topics covered:
- Large-scale batch processing workflows
- Distributed computing with multiple workers
- Resource monitoring and management
- Job queue management and scheduling
- Progress tracking and status reporting
- Error handling and recovery
- Processing optimization strategies
- Results aggregation and analysis
- Database integration for metadata
- Cloud processing capabilities

Author: SARPYX Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import json
import yaml
import sqlite3
import time
import psutil
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Process, cpu_count
from queue import Queue as ThreadQueue
import pickle
import hashlib
from datetime import datetime, timedelta
import shutil
import os
import signal
import sys

# SARPYX imports
from sarpyx.sla import SLAProcessor
from sarpyx.utils import io as sarpyx_io
from sarpyx.utils import viz as sarpyx_viz
from sarpyx.science import indices
from sarpyx.snapflow import engine as snap_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class JobConfig:
    """Configuration for individual processing jobs."""
    job_id: str
    input_path: str
    output_path: str
    processing_type: str = "standard"
    priority: int = 1
    max_retry: int = 3
    timeout_minutes: int = 60
    memory_limit_gb: float = 4.0
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'input_path': self.input_path,
            'output_path': self.output_path,
            'processing_type': self.processing_type,
            'priority': self.priority,
            'max_retry': self.max_retry,
            'timeout_minutes': self.timeout_minutes,
            'memory_limit_gb': self.memory_limit_gb,
            'custom_parameters': self.custom_parameters,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobConfig':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass
class JobStatus:
    """Status tracking for processing jobs."""
    job_id: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    worker_id: Optional[str] = None
    memory_usage_gb: float = 0.0
    processing_time_seconds: float = 0.0
    output_size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'worker_id': self.worker_id,
            'memory_usage_gb': self.memory_usage_gb,
            'processing_time_seconds': self.processing_time_seconds,
            'output_size_mb': self.output_size_mb
        }

class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.resources_data = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get system resources
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Get network I/O
                net_io = psutil.net_io_counters()
                
                # Record data point
                data_point = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3),
                    'network_bytes_sent': net_io.bytes_sent,
                    'network_bytes_recv': net_io.bytes_recv
                }
                
                self.resources_data.append(data_point)
                
                # Keep only recent data (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.resources_data = [
                    dp for dp in self.resources_data 
                    if dp['timestamp'] > cutoff_time
                ]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        if not self.resources_data:
            return {}
        
        latest = self.resources_data[-1]
        return {
            'cpu_percent': latest['cpu_percent'],
            'memory_percent': latest['memory_percent'],
            'memory_used_gb': latest['memory_used_gb'],
            'disk_percent': latest['disk_percent']
        }
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage."""
        if not self.resources_data:
            return {}
        
        return {
            'peak_cpu_percent': max(dp['cpu_percent'] for dp in self.resources_data),
            'peak_memory_percent': max(dp['memory_percent'] for dp in self.resources_data),
            'peak_memory_used_gb': max(dp['memory_used_gb'] for dp in self.resources_data)
        }
    
    def save_monitoring_data(self, output_path: str):
        """Save monitoring data to file."""
        import pandas as pd
        
        if self.resources_data:
            df = pd.DataFrame(self.resources_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Resource monitoring data saved to: {output_path}")

class JobDatabase:
    """Database for tracking job status and metadata."""
    
    def __init__(self, db_path: str = "batch_processing.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    processing_type TEXT,
                    priority INTEGER,
                    max_retry INTEGER,
                    timeout_minutes INTEGER,
                    memory_limit_gb REAL,
                    custom_parameters TEXT,
                    created_at TEXT
                )
            """)
            
            # Job status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_status (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL,
                    start_time TEXT,
                    end_time TEXT,
                    error_message TEXT,
                    retry_count INTEGER,
                    worker_id TEXT,
                    memory_usage_gb REAL,
                    processing_time_seconds REAL,
                    output_size_mb REAL,
                    FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                )
            """)
            
            # Processing statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                )
            """)
            
            conn.commit()
        
        logger.info(f"Job database initialized: {self.db_path}")
    
    def add_job(self, job_config: JobConfig):
        """Add a new job to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_config.job_id,
                job_config.input_path,
                job_config.output_path,
                job_config.processing_type,
                job_config.priority,
                job_config.max_retry,
                job_config.timeout_minutes,
                job_config.memory_limit_gb,
                json.dumps(job_config.custom_parameters),
                job_config.created_at.isoformat()
            ))
            
            # Initialize status
            status = JobStatus(job_config.job_id)
            self.update_job_status(status)
            
            conn.commit()
    
    def update_job_status(self, status: JobStatus):
        """Update job status in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO job_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                status.job_id,
                status.status,
                status.progress,
                status.start_time.isoformat() if status.start_time else None,
                status.end_time.isoformat() if status.end_time else None,
                status.error_message,
                status.retry_count,
                status.worker_id,
                status.memory_usage_gb,
                status.processing_time_seconds,
                status.output_size_mb
            ))
            
            conn.commit()
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM job_status WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            
            if row:
                status = JobStatus(
                    job_id=row[0],
                    status=row[1],
                    progress=row[2],
                    start_time=datetime.fromisoformat(row[3]) if row[3] else None,
                    end_time=datetime.fromisoformat(row[4]) if row[4] else None,
                    error_message=row[5],
                    retry_count=row[6],
                    worker_id=row[7],
                    memory_usage_gb=row[8],
                    processing_time_seconds=row[9],
                    output_size_mb=row[10]
                )
                return status
        
        return None
    
    def get_jobs_by_status(self, status: str) -> List[JobConfig]:
        """Get all jobs with specified status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT j.* FROM jobs j
                JOIN job_status js ON j.job_id = js.job_id
                WHERE js.status = ?
                ORDER BY j.priority DESC, j.created_at ASC
            """, (status,))
            
            jobs = []
            for row in cursor.fetchall():
                job_config = JobConfig(
                    job_id=row[0],
                    input_path=row[1],
                    output_path=row[2],
                    processing_type=row[3],
                    priority=row[4],
                    max_retry=row[5],
                    timeout_minutes=row[6],
                    memory_limit_gb=row[7],
                    custom_parameters=json.loads(row[8]) if row[8] else {},
                    created_at=datetime.fromisoformat(row[9])
                )
                jobs.append(job_config)
            
            return jobs
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count jobs by status
            cursor.execute("""
                SELECT status, COUNT(*) FROM job_status GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Get average processing time
            cursor.execute("""
                SELECT AVG(processing_time_seconds) FROM job_status 
                WHERE status = 'completed'
            """)
            avg_processing_time = cursor.fetchone()[0] or 0
            
            # Get total output size
            cursor.execute("""
                SELECT SUM(output_size_mb) FROM job_status 
                WHERE status = 'completed'
            """)
            total_output_size = cursor.fetchone()[0] or 0
            
            return {
                'status_counts': status_counts,
                'average_processing_time_seconds': avg_processing_time,
                'total_output_size_mb': total_output_size,
                'total_jobs': sum(status_counts.values())
            }

class WorkerProcess:
    """Individual worker process for job execution."""
    
    def __init__(self, worker_id: str, job_queue: Queue, result_queue: Queue, 
                 db_path: str, stop_event):
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.db_path = db_path
        self.stop_event = stop_event
        self.current_job = None
        
    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get next job from queue (with timeout)
                try:
                    job_data = self.job_queue.get(timeout=5.0)
                except:
                    continue
                
                if job_data is None:  # Shutdown signal
                    break
                
                # Process the job
                job_config = JobConfig.from_dict(job_data)
                self.current_job = job_config
                
                result = self._process_job(job_config)
                self.result_queue.put(result)
                
                self.current_job = None
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                if self.current_job:
                    error_result = {
                        'job_id': self.current_job.job_id,
                        'status': 'failed',
                        'error': str(e)
                    }
                    self.result_queue.put(error_result)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _process_job(self, job_config: JobConfig) -> Dict[str, Any]:
        """Process a single job."""
        start_time = time.time()
        
        # Update status to running
        db = JobDatabase(self.db_path)
        status = JobStatus(
            job_id=job_config.job_id,
            status="running",
            start_time=datetime.now(),
            worker_id=self.worker_id
        )
        db.update_job_status(status)
        
        try:
            # Initialize processor
            sla_processor = SLAProcessor()
            
            # Load data
            logger.info(f"Worker {self.worker_id}: Loading {job_config.input_path}")
            sar_data = sarpyx_io.load_sar_data(job_config.input_path)
            
            # Update progress
            status.progress = 0.2
            db.update_job_status(status)
            
            # Determine processing type and execute
            if job_config.processing_type == "sla":
                result = self._process_sla(sar_data, job_config, status, db)
            elif job_config.processing_type == "polarimetric":
                result = self._process_polarimetric(sar_data, job_config, status, db)
            elif job_config.processing_type == "interferometric":
                result = self._process_interferometric(sar_data, job_config, status, db)
            else:
                result = self._process_standard(sar_data, job_config, status, db)
            
            # Save results
            logger.info(f"Worker {self.worker_id}: Saving to {job_config.output_path}")
            self._save_results(result, job_config.output_path)
            
            # Calculate output size
            output_size_mb = 0
            if Path(job_config.output_path).exists():
                output_size_mb = Path(job_config.output_path).stat().st_size / (1024*1024)
            
            # Update final status
            processing_time = time.time() - start_time
            status.status = "completed"
            status.progress = 1.0
            status.end_time = datetime.now()
            status.processing_time_seconds = processing_time
            status.output_size_mb = output_size_mb
            status.memory_usage_gb = psutil.Process().memory_info().rss / (1024**3)
            
            db.update_job_status(status)
            
            return {
                'job_id': job_config.job_id,
                'status': 'completed',
                'processing_time': processing_time,
                'output_size_mb': output_size_mb
            }
            
        except Exception as e:
            # Update error status
            status.status = "failed"
            status.error_message = str(e)
            status.end_time = datetime.now()
            status.processing_time_seconds = time.time() - start_time
            
            db.update_job_status(status)
            
            logger.error(f"Worker {self.worker_id}: Job {job_config.job_id} failed: {e}")
            
            return {
                'job_id': job_config.job_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _process_sla(self, data: np.ndarray, job_config: JobConfig, 
                     status: JobStatus, db: JobDatabase) -> Dict[str, Any]:
        """Process SLA analysis."""
        sla_processor = SLAProcessor()
        
        # Configure parameters
        params = job_config.custom_parameters
        sub_apertures = params.get('sub_apertures', 4)
        overlap_factor = params.get('overlap_factor', 0.5)
        
        # Update progress
        status.progress = 0.4
        db.update_job_status(status)
        
        # Process sub-looks
        sla_results = sla_processor.process(
            data, 
            sub_apertures=sub_apertures,
            overlap_factor=overlap_factor
        )
        
        # Update progress
        status.progress = 0.8
        db.update_job_status(status)
        
        return sla_results
    
    def _process_polarimetric(self, data: np.ndarray, job_config: JobConfig,
                             status: JobStatus, db: JobDatabase) -> Dict[str, Any]:
        """Process polarimetric analysis."""
        # Simplified polarimetric processing
        status.progress = 0.5
        db.update_job_status(status)
        
        # Calculate polarimetric features
        if np.iscomplexobj(data):
            intensity = np.abs(data)**2
            phase = np.angle(data)
            
            results = {
                'intensity': intensity,
                'phase': phase,
                'polarimetric_features': {
                    'span': intensity,
                    'phase_variance': np.var(phase)
                }
            }
        else:
            results = {'intensity': data}
        
        status.progress = 0.9
        db.update_job_status(status)
        
        return results
    
    def _process_interferometric(self, data: np.ndarray, job_config: JobConfig,
                                status: JobStatus, db: JobDatabase) -> Dict[str, Any]:
        """Process interferometric analysis."""
        # Simplified interferometric processing
        status.progress = 0.6
        db.update_job_status(status)
        
        # Basic interferometric processing
        if np.iscomplexobj(data):
            # Calculate interferogram (simplified)
            master = data[:, :data.shape[1]//2]
            slave = data[:, data.shape[1]//2:]
            
            interferogram = master * np.conj(slave)
            coherence = np.abs(interferogram) / (np.abs(master) * np.abs(slave) + 1e-10)
            
            results = {
                'interferogram': interferogram,
                'coherence': coherence,
                'phase': np.angle(interferogram)
            }
        else:
            results = {'intensity': data}
        
        status.progress = 0.9
        db.update_job_status(status)
        
        return results
    
    def _process_standard(self, data: np.ndarray, job_config: JobConfig,
                         status: JobStatus, db: JobDatabase) -> Dict[str, Any]:
        """Process standard analysis."""
        # Basic processing
        status.progress = 0.6
        db.update_job_status(status)
        
        # Calculate basic statistics
        results = {
            'intensity': np.abs(data)**2 if np.iscomplexobj(data) else data,
            'statistics': {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
        }
        
        status.progress = 0.9
        db.update_job_status(status)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save processing results."""
        import h5py
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.h5':
            with h5py.File(output_path, 'w') as f:
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value, compression='gzip')
                    elif isinstance(value, dict):
                        group = f.create_group(key)
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, np.ndarray):
                                group.create_dataset(subkey, data=subvalue, compression='gzip')
                            else:
                                group.attrs[subkey] = subvalue
        else:
            # Save as pickle for other formats
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)

class BatchProcessor:
    """Main batch processing coordinator."""
    
    def __init__(self, max_workers: int = None, db_path: str = "batch_processing.db"):
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.db_path = db_path
        self.db = JobDatabase(db_path)
        self.resource_monitor = ResourceMonitor()
        
        # Multiprocessing components
        self.job_queue = None
        self.result_queue = None
        self.stop_event = None
        self.workers = []
        self.processes = []
        
        # Status tracking
        self.processing_active = False
        self.stats_thread = None
        
        logger.info(f"Batch processor initialized with {self.max_workers} workers")
    
    def add_job(self, input_path: str, output_path: str, processing_type: str = "standard",
                priority: int = 1, **kwargs) -> str:
        """Add a job to the processing queue."""
        job_id = self._generate_job_id(input_path)
        
        job_config = JobConfig(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            processing_type=processing_type,
            priority=priority,
            **kwargs
        )
        
        self.db.add_job(job_config)
        logger.info(f"Added job {job_id}: {input_path} -> {output_path}")
        
        return job_id
    
    def add_jobs_from_directory(self, input_dir: str, output_dir: str, 
                               file_pattern: str = "*.SAFE", 
                               processing_type: str = "standard") -> List[str]:
        """Add multiple jobs from a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find files matching pattern
        files = list(input_path.glob(file_pattern))
        job_ids = []
        
        for file_path in files:
            output_file = output_path / f"{file_path.stem}_processed.h5"
            job_id = self.add_job(
                str(file_path),
                str(output_file),
                processing_type=processing_type
            )
            job_ids.append(job_id)
        
        logger.info(f"Added {len(job_ids)} jobs from directory: {input_dir}")
        return job_ids
    
    def start_processing(self):
        """Start batch processing."""
        if self.processing_active:
            logger.warning("Processing already active")
            return
        
        logger.info("Starting batch processing...")
        
        # Initialize multiprocessing components
        manager = Manager()
        self.job_queue = manager.Queue()
        self.result_queue = manager.Queue()
        self.stop_event = manager.Event()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Queue pending jobs
        pending_jobs = self.db.get_jobs_by_status("pending")
        for job_config in pending_jobs:
            self.job_queue.put(job_config.to_dict())
        
        logger.info(f"Queued {len(pending_jobs)} pending jobs")
        
        # Start worker processes
        for i in range(self.max_workers):
            worker_id = f"worker_{i}"
            worker = WorkerProcess(
                worker_id, self.job_queue, self.result_queue, 
                self.db_path, self.stop_event
            )
            
            process = Process(target=worker.run)
            process.start()
            
            self.workers.append(worker)
            self.processes.append(process)
        
        # Start result processing thread
        self.processing_active = True
        self.stats_thread = threading.Thread(target=self._process_results)
        self.stats_thread.daemon = True
        self.stats_thread.start()
        
        logger.info(f"Started {self.max_workers} worker processes")
    
    def stop_processing(self, timeout: float = 30.0):
        """Stop batch processing."""
        if not self.processing_active:
            return
        
        logger.info("Stopping batch processing...")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Send shutdown signals
        for _ in range(self.max_workers):
            self.job_queue.put(None)
        
        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=timeout)
            if process.is_alive():
                logger.warning(f"Terminating unresponsive worker process")
                process.terminate()
                process.join(timeout=5.0)
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Stop result processing
        self.processing_active = False
        if self.stats_thread:
            self.stats_thread.join(timeout=5.0)
        
        self.workers.clear()
        self.processes.clear()
        
        logger.info("Batch processing stopped")
    
    def wait_for_completion(self, check_interval: float = 5.0):
        """Wait for all jobs to complete."""
        logger.info("Waiting for job completion...")
        
        while self.processing_active:
            summary = self.db.get_processing_summary()
            pending = summary['status_counts'].get('pending', 0)
            running = summary['status_counts'].get('running', 0)
            
            if pending == 0 and running == 0:
                logger.info("All jobs completed")
                break
            
            logger.info(f"Jobs remaining: {pending} pending, {running} running")
            time.sleep(check_interval)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current processing status summary."""
        summary = self.db.get_processing_summary()
        
        # Add resource usage
        current_resources = self.resource_monitor.get_current_usage()
        peak_resources = self.resource_monitor.get_peak_usage()
        
        summary.update({
            'current_resources': current_resources,
            'peak_resources': peak_resources,
            'active_workers': len([p for p in self.processes if p.is_alive()]),
            'processing_active': self.processing_active
        })
        
        return summary
    
    def cancel_job(self, job_id: str):
        """Cancel a specific job."""
        status = self.db.get_job_status(job_id)
        if status and status.status in ['pending', 'running']:
            status.status = 'cancelled'
            status.end_time = datetime.now()
            self.db.update_job_status(status)
            logger.info(f"Cancelled job: {job_id}")
    
    def retry_failed_jobs(self):
        """Retry all failed jobs."""
        failed_jobs = self.db.get_jobs_by_status("failed")
        
        for job_config in failed_jobs:
            status = self.db.get_job_status(job_config.job_id)
            if status.retry_count < job_config.max_retry:
                status.status = "pending"
                status.retry_count += 1
                status.error_message = None
                self.db.update_job_status(status)
                
                # Re-queue the job
                if self.processing_active:
                    self.job_queue.put(job_config.to_dict())
                
                logger.info(f"Retrying job {job_config.job_id} (attempt {status.retry_count})")
    
    def generate_processing_report(self, output_path: str = "processing_report.html"):
        """Generate comprehensive processing report."""
        summary = self.get_status_summary()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SARPYX Batch Processing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .completed {{ color: green; font-weight: bold; }}
                .failed {{ color: red; font-weight: bold; }}
                .running {{ color: blue; font-weight: bold; }}
                .pending {{ color: orange; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>SARPYX Batch Processing Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Processing Summary</h2>
                <div class="metric">
                    <strong>Total Jobs:</strong> {summary['total_jobs']}
                </div>
                <div class="metric">
                    <strong>Active Workers:</strong> {summary['active_workers']}
                </div>
                <div class="metric">
                    <strong>Processing Active:</strong> {'Yes' if summary['processing_active'] else 'No'}
                </div>
            </div>
            
            <div class="section">
                <h2>Job Status</h2>
                <table>
                    <tr><th>Status</th><th>Count</th></tr>
        """
        
        for status, count in summary['status_counts'].items():
            html_content += f'<tr><td class="{status}">{status.upper()}</td><td>{count}</td></tr>'
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metric">
                    <strong>Average Processing Time:</strong> {summary['average_processing_time_seconds']:.1f} seconds
                </div>
                <div class="metric">
                    <strong>Total Output Size:</strong> {summary['total_output_size_mb']:.1f} MB
                </div>
            </div>
            
            <div class="section">
                <h2>Resource Usage</h2>
                <table>
                    <tr><th>Resource</th><th>Current</th><th>Peak</th></tr>
        """
        
        current = summary.get('current_resources', {})
        peak = summary.get('peak_resources', {})
        
        for resource in ['cpu_percent', 'memory_percent']:
            current_val = current.get(resource, 0)
            peak_val = peak.get(f'peak_{resource}', 0)
            html_content += f'<tr><td>{resource.replace("_", " ").title()}</td><td>{current_val:.1f}%</td><td>{peak_val:.1f}%</td></tr>'
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Processing report generated: {output_path}")
        return output_path
    
    def _generate_job_id(self, input_path: str) -> str:
        """Generate unique job ID."""
        # Create hash from input path and timestamp
        content = f"{input_path}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _process_results(self):
        """Process results from worker processes."""
        while self.processing_active:
            try:
                # Check for results (with timeout)
                try:
                    result = self.result_queue.get(timeout=1.0)
                except:
                    continue
                
                # Log result
                job_id = result['job_id']
                status = result['status']
                
                if status == 'completed':
                    processing_time = result.get('processing_time', 0)
                    output_size = result.get('output_size_mb', 0)
                    logger.info(f"Job {job_id} completed in {processing_time:.1f}s, output: {output_size:.1f}MB")
                elif status == 'failed':
                    error = result.get('error', 'Unknown error')
                    logger.error(f"Job {job_id} failed: {error}")
                
            except Exception as e:
                logger.error(f"Error processing results: {e}")

def main():
    """
    Main function demonstrating batch processing capabilities.
    """
    print("SARPYX Batch Processing Demo")
    print("=" * 50)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(max_workers=2)
    
    # Example data paths (replace with actual data)
    data_dir = "data"
    output_dir = "batch_output"
    
    try:
        # Add jobs from directory
        print("\n1. Adding Jobs")
        print("-" * 20)
        
        # Add individual jobs for demonstration
        job_ids = []
        
        # Example job configurations
        sample_path = "data/S1A_S3_SLC__1SSH_20240621T052251_20240621T052319_054417_069F07_8466.SAFE"
        
        # SLA processing job
        job_id1 = batch_processor.add_job(
            input_path=sample_path,
            output_path=f"{output_dir}/sla_result.h5",
            processing_type="sla",
            priority=2,
            custom_parameters={'sub_apertures': 6, 'overlap_factor': 0.6}
        )
        job_ids.append(job_id1)
        
        # Polarimetric processing job
        job_id2 = batch_processor.add_job(
            input_path=sample_path,
            output_path=f"{output_dir}/polarimetric_result.h5",
            processing_type="polarimetric",
            priority=1
        )
        job_ids.append(job_id2)
        
        # Standard processing job
        job_id3 = batch_processor.add_job(
            input_path=sample_path,
            output_path=f"{output_dir}/standard_result.h5",
            processing_type="standard",
            priority=3
        )
        job_ids.append(job_id3)
        
        print(f"Added {len(job_ids)} jobs")
        
        # Start processing
        print("\n2. Starting Processing")
        print("-" * 20)
        
        batch_processor.start_processing()
        
        # Monitor progress
        print("\n3. Monitoring Progress")
        print("-" * 20)
        
        start_time = time.time()
        while True:
            summary = batch_processor.get_status_summary()
            
            pending = summary['status_counts'].get('pending', 0)
            running = summary['status_counts'].get('running', 0)
            completed = summary['status_counts'].get('completed', 0)
            failed = summary['status_counts'].get('failed', 0)
            
            print(f"Status - Pending: {pending}, Running: {running}, Completed: {completed}, Failed: {failed}")
            
            if pending == 0 and running == 0:
                break
            
            # Safety timeout for demo
            if time.time() - start_time > 300:  # 5 minutes
                print("Demo timeout reached")
                break
            
            time.sleep(10)
        
        # Generate final report
        print("\n4. Generating Report")
        print("-" * 20)
        
        report_path = batch_processor.generate_processing_report()
        print(f"Processing report: {report_path}")
        
        # Print final summary
        final_summary = batch_processor.get_status_summary()
        print(f"\nFinal Summary:")
        print(f"  Total jobs: {final_summary['total_jobs']}")
        print(f"  Completed: {final_summary['status_counts'].get('completed', 0)}")
        print(f"  Failed: {final_summary['status_counts'].get('failed', 0)}")
        print(f"  Average processing time: {final_summary['average_processing_time_seconds']:.1f}s")
        print(f"  Total output size: {final_summary['total_output_size_mb']:.1f}MB")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        print("\n5. Cleanup")
        print("-" * 20)
        
        batch_processor.stop_processing()
        print("Batch processing stopped")


if __name__ == "__main__":
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutdown signal received")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main()
