#!/usr/bin/env bash
set -euo pipefail

# Install Docker Engine on common Linux distributions.
# This script uses the official Docker repositories.

log() { echo "[docker-install] $*"; }

if [[ ${EUID} -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    exec sudo -E bash "$0" "$@"
  else
    echo "This script must be run as root (or with sudo)."
    exit 1
  fi
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot detect OS (missing /etc/os-release)."
  exit 1
fi

# shellcheck disable=SC1091
. /etc/os-release

OS_ID="${ID}"
OS_LIKE="${ID_LIKE:-}"
VERSION_CODE="${VERSION_CODENAME:-}"

normalize_repo_os() {
  case "${OS_ID}" in
    ubuntu|debian)
      echo "${OS_ID}";
      return 0
      ;;
    linuxmint|pop|elementary)
      echo "ubuntu";
      return 0
      ;;
  esac

  if [[ "${OS_LIKE}" == *"ubuntu"* ]]; then
    echo "ubuntu";
    return 0
  fi
  if [[ "${OS_LIKE}" == *"debian"* ]]; then
    echo "debian";
    return 0
  fi

  return 1
}

install_debian_like() {
  local repo_os
  repo_os="$(normalize_repo_os)" || {
    echo "Unsupported Debian-like distro: ${OS_ID}";
    exit 1
  }

  if [[ -z "${VERSION_CODE}" ]]; then
    # Some distros expose UBUNTU_CODENAME instead.
    VERSION_CODE="${UBUNTU_CODENAME:-}"
  fi
  if [[ -z "${VERSION_CODE}" ]]; then
    echo "Could not determine distribution codename.";
    exit 1
  fi

  log "Installing Docker for ${repo_os} (${VERSION_CODE})."

  apt-get update -y
  apt-get install -y ca-certificates curl gnupg

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${repo_os}/gpg" | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  local arch
  arch="$(dpkg --print-architecture)"
  {
    echo "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${repo_os} ${VERSION_CODE} stable"
  } > /etc/apt/sources.list.d/docker.list

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_rhel_like() {
  local pkg_mgr repo_os

  if command -v dnf >/dev/null 2>&1; then
    pkg_mgr="dnf"
  elif command -v yum >/dev/null 2>&1; then
    pkg_mgr="yum"
  else
    echo "No yum/dnf package manager found."
    exit 1
  fi

  case "${OS_ID}" in
    fedora)
      repo_os="fedora"
      ;;
    rhel)
      repo_os="rhel"
      ;;
    centos|rocky|almalinux)
      repo_os="centos"
      ;;
    *)
      if [[ "${OS_LIKE}" == *"rhel"* ]]; then
        repo_os="rhel"
      else
        repo_os="centos"
      fi
      ;;
  esac

  log "Installing Docker for ${repo_os} using ${pkg_mgr}."

  if [[ "${pkg_mgr}" == "dnf" ]]; then
    dnf -y install dnf-plugins-core
    dnf config-manager --add-repo "https://download.docker.com/linux/${repo_os}/docker-ce.repo"
    dnf -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  else
    yum -y install yum-utils
    yum-config-manager --add-repo "https://download.docker.com/linux/${repo_os}/docker-ce.repo"
    yum -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi
}

install_arch_like() {
  log "Installing Docker with pacman."
  pacman -Sy --needed --noconfirm docker
}

enable_docker() {
  if command -v systemctl >/dev/null 2>&1; then
    systemctl enable --now docker
  else
    log "systemctl not found; please start Docker manually."
  fi
}

add_user_to_group() {
  local user
  user="${SUDO_USER:-}"

  if [[ -z "${user}" && -n "${LOGNAME:-}" && "${LOGNAME}" != "root" ]]; then
    user="${LOGNAME}"
  fi

  if [[ -n "${user}" ]]; then
    if ! getent group docker >/dev/null 2>&1; then
      groupadd docker || true
    fi
    usermod -aG docker "${user}"
    log "Added ${user} to the docker group. Log out/in for group change to apply."
  fi
}

case "${OS_ID}" in
  ubuntu|debian|linuxmint|pop|elementary)
    install_debian_like
    ;;
  fedora|rhel|centos|rocky|almalinux)
    install_rhel_like
    ;;
  arch|manjaro)
    install_arch_like
    ;;
  *)
    if [[ "${OS_LIKE}" == *"debian"* || "${OS_LIKE}" == *"ubuntu"* ]]; then
      install_debian_like
    elif [[ "${OS_LIKE}" == *"rhel"* || "${OS_LIKE}" == *"fedora"* ]]; then
      install_rhel_like
    else
      echo "Unsupported distribution: ${OS_ID}"
      exit 1
    fi
    ;;
 esac

enable_docker
add_user_to_group

log "Docker installation complete."
log "Verify with: docker --version"
