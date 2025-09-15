# 0) Ensure you’re on the newest kernel and headers
sudo apt update
sudo apt -y full-upgrade
sudo apt -y install linux-headers-$(uname -r)

# 1) Reinstall + rebuild the exact server driver you chose (non-open variant shown)
sudo apt -y install --reinstall nvidia-kernel-common-565-server nvidia-dkms-565-server nvidia-driver-565-server

# (If you installed the open variant, replace with: nvidia-kernel-open-565-server nvidia-driver-565-server-open)

# 2) Check DKMS build status
sudo dkms status

# 3) If Secure Boot is ON, sign modules or disable it (see check below), then reboot
sudo reboot