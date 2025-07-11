#!/bin/bash
clear
export AWS_ENDPOINT_URL=https://eodata.dataspace.copernicus.eu/
aws s3 ls s3://eodata/Sentinel-1/SAR/
# aws s3 sync ./ s3://eodata// --request-payer requester


# Download 
aws s3 cp $1 $2
