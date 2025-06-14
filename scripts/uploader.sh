#!/bin/bash
# This script uploads the dataset to Hugging Face using the CLI.
# Ensure you have the Hugging Face CLI installed and authenticated

FOLDER="/Data_large/marine/PythonProjects/SAR/sarpyx/focused_data/"

echo "🚀 Starting dataset upload to Hugging Face..."
echo "📂 Dataset folder: $FOLDER"
echo "🔑 Make sure you are authenticated with Hugging Face CLI."

huggingface-cli upload-large-folder sirbastiano94/MAYA4 $FOLDER --repo-type=dataset

if [ $? -eq 0 ]; then
    echo "✅ Upload completed successfully!"
else
    echo "❌ Upload failed. Please check the error messages above."
fi