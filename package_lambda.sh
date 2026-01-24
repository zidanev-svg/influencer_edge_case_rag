#!/bin/bash

echo "Packaging Lambda function..."

mkdir lambda_package
pip install -r lambda_requirements.txt -t lambda_package/
cp lambda_handler.py lambda_package/
cd lambda_package
zip -r ../lambda_function.zip .
cd ..
rm -rf lambda_package

echo ""
echo "✅ Lambda package created: lambda_function.zip"
echo ""
echo "Next: Follow instructions in LAMBDA_INSTRUCTIONS.md to deploy"
