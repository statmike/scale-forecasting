# Notebook fan-out — diagnostic (temporary, delete after)

The 400s were caused by the URL wrapping across a line on paste. These blocks put the
long URL and JSON body into variables/files so nothing wraps mid-token. Paste each
fenced block as a whole.

## 0. Set the environment (paste as one block)

```bash
PROJECT=gcp-scale-forecasting
REGION=us-central1
cd ~/scale-forecasting
RUNNER_SA="scale-forecasting-runner@$PROJECT.iam.gserviceaccount.com"
CODE_BUCKET="$PROJECT-code"
MAIN_TEMPLATE="projects/$PROJECT/locations/$REGION/notebookRuntimeTemplates/4277766536095072256"
SPARK_TEMPLATE="projects/$PROJECT/locations/$REGION/notebookRuntimeTemplates/197505273697402880"
echo "SA=[$RUNNER_SA]"
echo "BUCKET=[$CODE_BUCKET]"
echo "MAIN=[$MAIN_TEMPLATE]"
echo "SPARK=[$SPARK_TEMPLATE]"
```

## 1. Single diagnostic POST (paste as one block)

```bash
TOKEN=$(gcloud auth print-access-token)
NB=$(base64 -w0 notebooks/07_scale_review.ipynb)
URL="https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$REGION/notebookExecutionJobs"
cat > /tmp/nbjob.json <<EOF
{"displayName":"diag-test","directNotebookSource":{"content":"$NB"},"notebookRuntimeTemplateResourceName":"$MAIN_TEMPLATE","gcsOutputUri":"gs://$CODE_BUCKET/notebooks","serviceAccount":"$RUNNER_SA","executionTimeout":"900s"}
EOF
curl -sS -w "\n---HTTP_STATUS:%{http_code}---\n" -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @/tmp/nbjob.json
```

Read the result:
- `HTTP_STATUS:200` (or 201) + JSON with `"name": "...notebookExecutionJobs/..."` -> it works. Go to step 2.
- `HTTP_STATUS:400` + error body -> paste the body; a named field is bad.
- `HTTP_STATUS:403` -> runner SA missing a permission on this project.

## 2. Real fan-out — all 8 notebooks (paste as one block)

Only run this after step 1 returns 200.

```bash
uv run python -m scale_forecasting.notebook_acceptance \
  --no-wait --tier full \
  --project "$PROJECT" --region "$REGION" \
  --main-template "$MAIN_TEMPLATE" --spark-template "$SPARK_TEMPLATE" \
  --service-account "$RUNNER_SA" \
  --gcs-output "gs://$CODE_BUCKET/notebooks" \
  --run-label "demo-$(date +%Y%m%d)"
```

Watch them render here:
https://console.cloud.google.com/vertex-ai/colab/execution-jobs?project=gcp-scale-forecasting
