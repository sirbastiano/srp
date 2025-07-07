#!/bin/bash

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version

aws configure

# AWS Access Key ID [None]: <access_key>
# AWS Secret Key ID [None]: <secret_key>
# Default region name [None]: default
# Default output format [None]: text

