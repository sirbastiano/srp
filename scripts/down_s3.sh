#!/bin/bash

export AWS_ENDPOINT_URL=https://eodata.dataspace.copernicus.eu/
aws s3 ls s3://eodata/
# aws s3 sync ./ s3://eodata// --request-payer requester