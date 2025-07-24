#bin/bash

# Clean up Docker images
echo "Cleaning up Docker images..."
docker image prune -a -f
# Pull the latest Docker image from Docker Hub
echo "Pulling the latest Docker image from Docker Hub..."
docker pull sirbastiano94/sarpyx:latest

# Run the Docker container with the specified image
echo "Running the Docker container..."
nohup docker run --rm -it irbastiano94/sarpyx

# ===============================
cat > .s5cfg <<EOF
[default]
aws_access_key_id = FMO7SZ7AFIY8PPB7XF5D
aws_secret_access_key = viauXvqsXL2uNmVzuQPjszB4Y8VV4POKNtet5pHG
aws_region = eu-central-1
host_base = eodata.dataspace.copernicus.eu
host_bucket = eodata.dataspace.copernicus.eu
use_https = true
check_ssl_certificate = true
EOF


./scripts/pipeline/single.sh S1A_S5_RAW__0SDV_20240613T185333_20240613T185404_054308_069B3E_61BF.SAFE
# ================================
