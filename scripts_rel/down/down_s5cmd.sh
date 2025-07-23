#!/bin/bash
clear
export AWS_ENDPOINT_URL=https://eodata.dataspace.copernicus.eu/
# aws s3 ls s3://eodata/Sentinel-1/SAR/
# aws s3 sync ./ s3://eodata// --request-payer requester

s5cmd --endpoint-url $AWS_ENDPOINT_URL --credentials-file /Data_large/marine/PythonProjects/SAR/sarpyx/.s5cfg ls s3://eodata/Sentinel-1/SAR/

# Download 
s5cmd --credentials-file /Data_large/marine/PythonProjects/SAR/sarpyx/.s5cfg cp $1 $2
