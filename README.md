# Influencer Edge Case RAG

This is the first version of the deal lookup system, built in January 2026. It focused on edge cases (unusually high or low priced deals) and on getting the lookup running as an AWS function.

The later, more complete work lives in a separate repo (`pinecone_upload`). This repo is kept as the original prototype.

---

## What it does

- Stores example deals in a searchable database.
- Runs a lookup that takes a deal and returns similar past deals.
- Packages that lookup so it can run as an AWS Lambda function.

---

## Files

- `upload_to_pinecone.py` — loads the example deals into the database
- `query_edge_cases.py` — runs lookups against the database
- `lambda_handler.py` — the lookup packaged as an AWS function
- `package_lambda.sh` — builds the deployment package
- `lambda_function.zip` — the built package
- `test_lambda.py` — tests for the function
- `mock_data/` — example deals used for testing (high and low price cases)
- `LAMBDA_INSTRUCTIONS.md` — how to deploy the function

---

## Setup

1. Fill in the settings file (`.env`) with the required keys.
2. Install the packages:

   ```
   pip install -r lambda_requirements.txt
   ```

---

## How this relates to the newer work

- This repo — the first prototype. Edge cases and the AWS function.
- `pinecone_upload` — the full system that came after. Real deal data, the weighted matching upgrade, and the contract review.

If you are looking for the current system, use `pinecone_upload`.

---

## Note

The `.env` file holds private keys. It is never committed and is already listed in `.gitignore`.
