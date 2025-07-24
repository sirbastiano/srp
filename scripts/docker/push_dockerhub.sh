#!/bin/bash
# Tag and push the Docker image to Docker Hub
dockerhub_username="sirbastiano94"  # Replace with your Docker Hub username
local_image_name="sarpyx-s11"
remote_image_name="sarpyx"
tag="latest"

# Check if user is logged in to Docker Hub
echo "Checking Docker Hub authentication..."
if ! docker info | grep -q "Username: ${dockerhub_username}"; then
    echo "Not logged in to Docker Hub. Please login:"
    docker login
    
    # Check if login was successful
    if [ $? -ne 0 ]; then
        echo "Failed to login to Docker Hub."
        exit 1
    fi
fi

# Tag the existing local image with the correct Docker Hub naming convention
echo "Tagging local image ${local_image_name}:${tag} as ${dockerhub_username}/${remote_image_name}:${tag}..."
docker tag ${local_image_name}:${tag} ${dockerhub_username}/${remote_image_name}:${tag}

# Check if tagging was successful
if [ $? -eq 0 ]; then
    echo "Tagging successful. Pushing to Docker Hub..."
    docker push ${dockerhub_username}/${remote_image_name}:${tag}
    
    if [ $? -eq 0 ]; then
        echo "Successfully pushed ${dockerhub_username}/${remote_image_name}:${tag} to Docker Hub!"
    else
        echo "Failed to push to Docker Hub."
        exit 1
    fi
else
    echo "Failed to tag the image."
    exit 1
fi