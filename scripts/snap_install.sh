#!/bin/bash
# Install SNAP (Sentinel Application Platform) on a Linux system
# This script downloads the SNAP installer and runs it silently

wget -q https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-12.0.0.sh -O /tmp/snap_installer.sh
chmod +x /tmp/snap_installer.sh 
/tmp/snap_installer.sh -q -dir /usr/local/snap 
rm /tmp/snap_installer.sh

echo 'export PATH="${PATH}:/home/vessel/esa-snap/bin"' >> ~/.bashrc
source ~/.bashrc