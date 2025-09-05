
# Sentinel-1 Metadata Parameters

Sentinel-1 is a satellite mission developed by the European Space Agency (ESA) to provide all-weather, day-and-night imaging of Earth's surface. The mission uses Synthetic Aperture Radar (SAR) to acquire data, which is processed and analyzed to generate images and other geospatial products. The SAR data is stored in a raw format, which includes various parameters that provide information about the data and how it was acquired. These parameters include the Packet Version Number, Packet Type, Coarse Time, Fine Time, Calibration Mode, Signal Type, and others. Understanding these parameters is essential for processing and analyzing the raw data and generating useful geospatial products.

<hr>

<p style="font-size: 11px; line-height: 1.3;">For a more detailed description, please refer to: https://sentinel.esa.int/documents/247904/0/Sentinel-1-SAR-Space-Packet-Protocol-Data-Unit.pdf/d47f3009-a37a-43f9-8b65-da858f6fb1ca*</p>


<hr>

# Ancillary Information

<table>
  <tr>
    <th>Element Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Packet Version Number</td>
    <td>This parameter denotes the version number of the data packet structure, ensuring the correct interpretation of the packet content. The version number is a critical aspect in maintaining backward compatibility and adapting to any future revisions in the data format.</td>
  </tr>
  <tr>
    <td>Packet Type</td>
    <td>The Packet Type is a flag that indicates whether the data packet contains telemetry or telecommand data. Telemetry data refers to the data transmitted from the satellite to the ground station, while telecommand data are the instructions sent from the ground station to the satellite.</td>
  </tr>
  <tr>
    <td>Secondary Header Flag</td>
    <td>This flag signifies the presence or absence of a secondary header in the data packet. Secondary headers provide additional information about the packet's content and its processing requirements, crucial for the accurate handling and interpretation of the data.</td>
  </tr>
  <tr>
    <td>PID</td>
    <td>The Packet Identification (PID) is a unique identifier assigned to each data packet, allowing for clear distinction and tracking of the individual packets. The PID is crucial in ensuring that the correct data packets are processed, avoiding potential errors or data loss during data transmission, processing, and analysis.</td>
  </tr>
  <tr>
    <td>PCAT</td>
    <td>The Packet Category (PCAT) is a parameter that classifies the data packets into different categories based on their content or function. This classification system enables efficient handling and processing of the data packets by distinguishing between various data types and processing requirements. The PCAT simplifies the data management process by ensuring that each packet is routed to the appropriate processing module.</td>
  </tr>
  <tr>
    <td>Sequence Flags</td>
    <td>The Sequence Flags are a set of indicators that describe the sequencing status of the data packets within a specific data stream. These flags provide information on whether a packet is the first, last, or an intermediate packet in a sequence. This information is crucial for ensuring the proper ordering and concatenation of data packets during data processing and analysis.</td>
  </tr>
  <tr>
    <td>Packet Sequence Count</td>
    <td>The Packet Sequence Count is a numeric value that increments for each successive data packet in a sequence. This parameter allows for tracking the order of data packets and identifying any missing or out-of-sequence packets during transmission or processing. It is essential for maintaining data integrity and ensuring the accurate reconstruction of the original data stream.</td>
  </tr>
  <tr>
    <td>Packet Data Length</td>
    <td>This element represents the total length of the data field within a packet, measured in bytes. It provides crucial information for accurately decoding and processing the data payload of the packet. The Packet Data Length enables the correct allocation of memory resources during data handling, ensuring efficient and error-free data extraction and processing.</td>
  </tr>
  <tr>
    <td>Coarse Time</td>
    <td>The Coarse Time is a parameter that indicates the time at which a data packet was generated, typically expressed as the number of seconds elapsed since a reference epoch (e.g., GPS time or UTC). This time stamp is essential for synchronizing and aligning the data packets with other data sources and for establishing a temporal context for the remote sensing observations.</td>
  </tr>
  <tr>
    <td>Fine Time</td>
    <td>The Fine Time parameter provides a high-resolution time stamp for a data packet, often expressed in fractions of a second. This element complements the Coarse Time, allowing for precise time alignment and synchronization of the data packets. The combined use of Coarse Time and Fine Time ensures that the remote sensing observations are accurately correlated with other data sources and time-referenced events.</td>
  </tr>
  <tr>
    <td>Sync</td>
    <td>The Sync element is a specific pattern or sequence of bits within a data packet that enables the identification of packet boundaries and the synchronization of data streams. The Sync Marker represents a bit pattern to support (re-)synchronisation of packet data on Space Packet layer level (e.g. in case of corruptions or disruptions in a continuous stream of Space Packets).</td>
  </tr>
  <tr>
    <td>Data Take ID</td>
    <td>The Data Take ID is a unique identifier assigned to a specific data acquisition event, also known as a data take. This identifier allows for the clear distinction and tracking of individual data takes, enabling efficient organization and management of the remote sensing observations. The Data Take ID is essential for correlating different data sets, tracking data provenance, and facilitating subsequent processing and analysis tasks. The Data Take ID is supposed to support ground operations to track the E2E life cycle of a data take from the planning, commanding up to the downlinking and reception of the related Space Packets of the data take. The Data Take ID will be uplinked as part of the “Perform Measurement” and “Perform Test” TC. Selection of the Data Take ID is under ground control.</td>
  </tr>
  <tr>
    <td>ECC Number</td>
    <td>The Error Control Code (ECC) Number identifies the selected Measurement, Test or RF Characterisation mode</td>
      </tr>
  <tr>
    <td>Test Mode</td>
    <td>The Test Mode is a flag indicating whether the satellite is operating in normal data acquisition mode or in a specific test mode. Test modes are typically used for system performance evaluations, calibration, or troubleshooting activities. This flag is crucial for distinguishing between operational and test data, ensuring that the data processing and analysis tasks are performed using the appropriate data sets.</td>
  </tr>
  <tr>
    <td>Rx Channel ID</td>
    <td>The Rx Channel ID identifies the Rx channel generating the packet data. Rx polarisation is and Rx channel are in fixed relation. Therefore, the Rx Channel ID also identifies the Rx polarisation of the channel (RxV-Pol. or RxH-Pol.)</td>
  </tr>
  <tr>
    <td>Instrument Configuration ID</td>
    <td>The Instrument Configuration ID is intended to support ground operations. It identifies in the Space Packets the onboard configuration of the Instrument under which the Instrument has operated and generated the data take. Knowledge of the configuration is a prerequisite for ground processing of the data take raw data.
    The Instrument configuration ID is a patchable Instrument parameter and is under control of ground operations. It has to be patched together with an Instrument configuration change. An Instrument configuration change is mainly induced by a change of the onboard Radar Data Base (RDB), e.g. change of beam tables, ECC programs, etc...</td>
  </tr>
  <tr>
    <td>Sub-commutated Ancillary Data Word Index</td>
    <td>The Sub-commutated Ancilliary Data Word Index is a parameter that indicates the position of a specific ancillary data word within a sub-commutation cycle. Sub-commutation is a technique used to organize and transmit ancillary data alongside the primary data payload. This index is essential for the correct extraction and interpretation of the ancillary data, ensuring that the relevant metadata is accurately associated with the remote sensing observations.</td>
  </tr>
  <tr>
    <td>Sub-commutated Ancillary Data Word</td>
    <td>The Sub-commutated Ancilliary Data Word is the actual content of an ancillary data word within the sub-commutation cycle. This data typically includes metadata, calibration information, or other supplementary data necessary for the accurate processing and analysis of the primary remote sensing observations. The Sub-commutated Ancilliary Data Word ensures that the essential ancillary information is readily available for data handling tasks.</td>
  </tr>
  <tr>
    <td>Space Packet Count</td>
    <td>The Space Packet Count is a numeric value that keeps track of the total number of space packets transmitted or received within a specific data sequence. This parameter is crucial for monitoring data transmission, ensuring data integrity, and detecting potential data loss or corruption. The Space Packet Count aids in the efficient organization and management of the remote sensing data.</td>
  </tr>
  <tr>
    <td>PRI Count</td>
    <td>The Pulse Repetition Interval (PRI) Count is a parameter that represents the number of radar pulses transmitted during a specific data acquisition event. This information is vital for accurately interpreting and processing the remote sensing observations, as the PRI is an essential aspect of the radar system's operation. The PRI Count ensures that the correct temporal and spatial context is established for the acquired data.</td>
  </tr>
  <tr>
    <td>Error Flag</td>
    <td>The Error Flag is an indicator that signals the presence of errors or anomalies in the data packet, such as transmission errors, data corruption, or instrument malfunctions. This flag is essential for identifying and handling erroneous data, ensuring that the remote sensing observations are accurate, reliable, and suitable for further processing and analysis.</td>
  </tr>
  <tr>
    <td>BAQ Mode</td>
    <td>The Block Adaptive Quantization (BAQ) Mode is a parameter that specifies the quantization method applied to the radar data. BAQ is a data compression technique used to optimize the storage and transmission of remote sensing data while minimizing the loss of information. The BAQ Mode is essential for ensuring that the data is accurately decoded and reconstructed during processing and analysis tasks.</td>
  </tr>
  <tr>
    <td>BAQ Block Length</td>
    <td>The BAQ Block Length is a parameter that defines the length of the data blocks used in the BAQ process. This length is critical for correctly partitioning and processing the compressed data, ensuring that the data is accurately decoded and reconstructed. The BAQ Block Length is an essential aspect of the BAQ technique, affecting the efficiency and quality of the data compression process.</td>
  </tr>
  <tr>
    <td>Range Decimation</td>
    <td>The Range Decimation is a parameter that indicates the level of range data decimation applied during data acquisition. Range decimation is a technique used to reduce the volume of range data by selectively discarding or averaging range samples. This parameter is crucial for accurately interpreting and processing the remote sensing observations, as it affects the spatial resolution and sampling characteristics of the acquired data.</td>
  </tr>
  <tr>
    <td>Rx Gain</td>
    <td>The Reception (Rx) Gain is a parameter that represents the gain setting applied to the radar receiver during data acquisition. This setting affects the sensitivity and dynamic range of the receiver, influencing the quality and accuracy of the remote sensing observations. The Rx Gain is essential for properly calibrating and processing the acquired data.</td>
  </tr>
  <tr>
    <td>Tx Ramp Rate</td>
    <td>The Transmission (Tx) Ramp Rate is a parameter that defines the rate of change in the frequency of the transmitted radar pulse, typically expressed in MHz/µs. This rate affects the pulse's frequency modulation and is an essential aspect of the radar system's operation. The Tx Ramp Rate is crucial for accurately interpreting and processing the remote sensing observations.</td>
  </tr>
  <tr>
    <td>Tx Pulse Start Frequency</td>
    <td>The Transmission (Tx) Pulse Start Frequency is a parameter that specifies the starting frequency of the transmitted radar pulse. This frequency is a critical component of the radar system's operation and affects the radar's range resolution and sensing capabilities. The Tx Pulse Start Frequency is essential for the accurate processing and analysis of the remote sensing data.</td>
  </tr>
  <tr>
    <td>Tx Pulse Length</td>
    <td>The Transmission (Tx) Pulse Length is a parameter that defines the duration of the transmitted radar pulse, typically expressed in microseconds. This pulse length affects the radar's range resolution and energy distribution, influencing the quality and accuracy of the remote sensing observations. The Tx Pulse Length is a critical aspect of the radar system's operation, ensuring the correct spatial context for the acquired data.
    </td>
  </tr>
  <tr>
    <td>Rank</td>
    <td>The Rank is a parameter that represents the hierarchical organization or ordering of data within a specific data set or processing stage. This parameter is essential for managing and organizing the data, ensuring that the correct data is used for specific processing tasks or analysis requirements. The Rank parameter helps maintain the integrity and consistency of the remote sensing observations.</td>
  </tr>
  <tr>
    <td>PRI</td>
    <td>The Pulse Repetition Interval (PRI) is a parameter representing the time interval between successive radar pulses, typically expressed in microseconds. The PRI affects the radar system's operation, including range coverage and Doppler processing. This parameter is crucial for accurately interpreting and processing the remote sensing observations.</td>
  </tr>
  <tr>
    <td>SWST</td>
    <td>The Sliding Window Start Time (SWST) is a parameter that defines the start time of a sliding window used for range data processing. The SWST is essential for managing range data organization and processing, affecting the spatial resolution and sampling characteristics of the acquired data.</td>
  </tr>
  <tr>
    <td>SWL</td>
    <td>The Sliding Window Length (SWL) is a parameter that represents the duration of the sliding window used for range data processing. This parameter affects the range resolution and data processing requirements, and it is crucial for accurately interpreting and processing the remote sensing observations.</td>
  </tr>
  <tr>
    <td>SAS SSB Flag</td>
    <td>SAS stands for "SAR Auxiliary Subsystem," which is a part of the SAR processing system responsible for auxiliary data management. The auxiliary data are essential for precise SAR image processing and geolocation. SSB is an abbreviation for "Single Sideband," which refers to a technique used in radio communications to reduce the bandwidth required to transmit a signal. The "SAS SSB Flag" is a metadata attribute included in the Sentinel-1 RAW product files, indicating the status of the single sideband processing applied to the SAR data. The flag can be set to "True" or "False." If it is set to "True," it means the single sideband processing has been applied correctly. If it is set to "False," there may be issues in the SAR data processing or quality, and users should be cautious when using the data for analysis or applications.Users of Sentinel-1 RAW data can check the metadata of the product files to determine the status of the "SAS SSB Flag" and take necessary actions if the flag indicates potential issues with the data.</td>
  </tr>
  <tr>
    <td>Polarisation</td>
    <td>The Polarisation is a parameter that indicates the polarization configuration of the transmitted and received radar signals. Polarization can be horizontal (H) or vertical (V) and affects the radar system's sensitivity to different surface properties and scattering mechanisms. The Polarisation parameter is essential for accurate data interpretation and processing.</td>
  </tr>
  <tr>
    <td>Temperature Compensation</td>
    <td>The Temperature Compensation is a parameter that accounts for the effects of temperature variations on the radar system's performance. This compensation is essential for maintaining the accuracy, reliability, and consistency of the remote sensing observations, as temperature changes can influence the radar system's characteristics, including gain, noise figure, and frequency response.</td>
  </tr>
  <tr>
    <td>Calibration Mode</td>
    <td>The Calibration Mode is a flag indicating whether the satellite is operating in normal data acquisition mode or in a specific calibration mode. Calibration modes are typically used for system performance evaluations, calibration activities, or troubleshooting tasks. This flag is crucial for distinguishing between operational and calibration data, ensuring that the data processing and analysis tasks are performed using the appropriate data sets.</td>
  </tr>
  <tr>
    <td>Tx Pulse Number</td>
    <td>The Transmission (Tx) Pulse Number is a parameter that represents the unique identifier for each transmitted radar pulse within a data acquisition sequence. This number is essential for organizing and tracking radar pulses, ensuring the correct temporal and spatial context for the remote sensing observations, and facilitating accurate data processing and analysis.</td>
  </tr>
  <tr>
    <td>Signal Type</td>
    <td>The Signal Type is a parameter that indicates the type of radar signal used in the data acquisition process, such as chirp or pulse. This information is vital for accurately interpreting and processing the remote sensing observations, as different signal types have unique characteristics and require specific processing techniques. The Signal Type parameter ensures that the appropriate processing methods are applied to the acquired data.</td>
  </tr>
  <tr>
    <td>Swap Flag</td>
    <td>The Swap Flag is an indicator that specifies whether a data swap operation has been performed on the data packet. Data swaps can be used for various purposes, such as reordering data samples, correcting data alignment, or optimizing data organization. This flag is essential for ensuring the correct interpretation and processing of the remote sensing observations, as it provides information on the organization and structure of the data.</td>
  </tr>
  <tr>
    <td>Swath Number</td>
    <td>The Swath Number is a parameter that represents the unique identifier for each swath within a data acquisition event. Swaths are segments of the Earth's surface imaged by the radar system during a single pass. The Swath Number is crucial for organizing and tracking the remote sensing observations, ensuring the correct spatial context for the data, and facilitating accurate data processing and analysis.</td>
  </tr>
  <tr>
    <td>Number of Quads</td>
    <td>The Number of Quads is a parameter that defines the number of quadrature data pairs (I/Q samples) in a radar data packet. Quadrature data is used to represent both amplitude and phase information in radar signals. The Number of Quads parameter is essential for managing data organization, ensuring that the correct number of data samples are processed, and maintaining the integrity of the remote sensing observations during processing and analysis.</td>
  </tr>
</table>