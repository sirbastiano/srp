#!/bin/sh

# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/30539778/Install+SNAP+on+the+command+line
# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/30539785/Update+SNAP+from+the+command+line
# http://step.esa.int/main/download/snap-download/

SNAPVER=12
# avoid NullPointer crash during S-1 processing
java_max_mem=10G

# install and update snap
wget -q -O /src/snap/esa-snap_all_unix_${SNAPVER}_0_0.sh \
  "http://step.esa.int/downloads/${SNAPVER}.0/installers/esa-snap_all_unix_${SNAPVER}_0_0.sh"

# # hack to make it run on alpine
sh /src/snap/esa-snap_all_unix_${SNAPVER}_0_0.sh -q -varfile /src/snap/response.varfile

# one more hack to keep using system java
sed -i 's+jdkhome="./jre"+jdkhome="$JAVA_HOME"+g' /usr/local/snap/etc/snap.conf
# freezing, when no updates available. Now there are updates, so reactivating.
/usr/local/snap/bin/snap --nosplash --nogui --modules --update-all
rm -rf /usr/local/snap/jre

# cleanup installer
rm -f /src/snap/esa-snap_all_unix_${SNAPVER}_0.sh

################################################################################
# keep for debugging
# export INSTALL4J_KEEP_TEMP=yes