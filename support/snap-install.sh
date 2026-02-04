#!/bin/sh
# DISCLAIMER: This script is provided as-is without any guarantees or warranty.
# The user assumes all responsibility and risk for the use of this script.
# The authors are not liable for any damage or data loss arising from its use.
# This script works for linux.


# Set SNAP version to install (12 or 13)[IMPORTANT]
VERSION=13

# Links: 
# SNAP 12: https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-12.0.0.sh
# SNAP 13: https://download.esa.int/step/snap/13.0/installers/esa-snap_sentinel_linux-13.0.0.sh
# SNAP 13MacOS-Intel: https://download.esa.int/step/snap/13.0/installers/esa-snap_sentinel_macos_intel-13.0.0.dmg
# SNAP 13MacOS-ARM: https://download.esa.int/step/snap/13.0/installers/esa-snap_sentinel_macos_arm-13.0.0.dmg
# SNAP 13Windows: https://download.esa.int/step/snap/13.0/installers/esa-snap_sentinel_windows-13.0.0.exe

# get filepath of current file 
CURRENT_FILE_PATH="$(realpath "$0")"
# go up two directories
BASE_DIR="$(dirname "$(dirname "$(dirname "$CURRENT_FILE_PATH")")")"


SNAP_DIR="${BASE_DIR}/snap${VERSION}"

echo "Installing SNAP version $VERSION..."
# Install required packages
echo 'Installing packages for S1 data processing...'
sudo apt update
sudo apt-get install -y libfftw3-dev libtiff5-dev gdal-bin gfortran libgfortran5 jblas git curl --fix-missing 

if [ "$VERSION" = "12" ]; then
    # VERSION 12 installation
    echo 'Configuring SNAP 12 installation...'
    curl -O https://download.esa.int/step/snap/13.0/installers/esa-snap_sentinel_linux-13.0.0.sh
    chmod +x esa-snap_all_linux-12.0.0.sh
    echo -e "deleteAllSnapEngineDir\$Boolean=false\ndeleteOnlySnapDesktopDir\$Boolean=false\nexecuteLauncherWithPythonAction\$Boolean=false\nforcePython\$Boolean=false\npythonExecutable=/usr/bin/python\nsys.adminRights\$Boolean=true\nsys.component.RSTB\$Boolean=true\nsys.component.S1TBX\$Boolean=true\nsys.component.S2TBX\$Boolean=false\nsys.component.S3TBX\$Boolean=false\nsys.component.SNAP\$Boolean=true\nsys.installationDir=$(pwd)/snap\nsys.languageId=en\nsys.programGroupDisabled\$Boolean=false\nsys.symlinkDir=/usr/local/bin" > snap.varfile
    echo 'Installing SNAP 12...'
    ./esa-snap_all_linux-12.0.0.sh -q -varfile "$(pwd)/snap.varfile" -dir "${SNAP_DIR}"
    echo 'Configuring SNAP memory settings...'
    echo "-Xmx8G" > "${SNAP_DIR}/snap/bin/gpt.vmoptions"
    echo 'SNAP 12 installation complete.'

elif [ "$VERSION" = "13" ]; then
    # VERSION 13 installation
    echo 'Configuring SNAP 13 installation...'
    curl -O https://download.esa.int/step/snap/13.0/installers/esa-snap_sentinel_linux-13.0.0.sh
    chmod +x esa-snap_sentinel_linux-13.0.0.sh
    echo -e "deleteAllSnapEngineDir\$Boolean=false\ndeleteOnlySnapDesktopDir\$Boolean=false\nexecuteLauncherWithPythonAction\$Boolean=false\nforcePython\$Boolean=false\npythonExecutable=/usr/bin/python\nsys.adminRights\$Boolean=true\nsys.component.RSTB\$Boolean=true\nsys.component.S1TBX\$Boolean=true\nsys.component.S2TBX\$Boolean=false\nsys.component.S3TBX\$Boolean=false\nsys.component.SNAP\$Boolean=true\nsys.installationDir=${SNAP_DIR}\nsys.languageId=en\nsys.programGroupDisabled\$Boolean=false\nsys.symlinkDir=/usr/local/bin" > snap.varfile
    echo 'Installing SNAP 13...'
    ./esa-snap_sentinel_linux-13.0.0.sh -q -varfile "$(pwd)/snap.varfile" -dir "${SNAP_DIR}"
    echo 'Configuring SNAP memory settings...'
    echo "-Xmx16G" > "${SNAP_DIR}/snap/bin/gpt.vmoptions"
    echo 'SNAP 13 installation complete.'

else
    echo "Error: Invalid VERSION. Please set VERSION to either 12 or 13."
    exit 1
fi