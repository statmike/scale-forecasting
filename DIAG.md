# Notebook fan-out — diagnostic + curl fan-out (temporary, delete after)

Two fixes were needed:
1. Template must be the **full** resource path, not the bare numeric id.
2. Personal (non-corp-org) projects require the request to carry
   `labels: { "aiplatform.googleapis.com/notebook_runtime_out_of_org_warning": "ack" }`
   or Vertex rejects the submit with FAILED_PRECONDITION.

The blocks below build the long URL/JSON into variables/files so nothing wraps on paste,
and include the ack label. Paste each fenced block as a whole.

## 0. Set the environment (paste as one block)

```bash
PROJECT=gcp-scale-forecasting
REGION=us-central1
cd ~/scale-forecasting
RUNNER_SA="scale-forecasting-runner@$PROJECT.iam.gserviceaccount.com"
CODE_BUCKET="$PROJECT-code"
MAIN_TEMPLATE="projects/$PROJECT/locations/$REGION/notebookRuntimeTemplates/4277766536095072256"
SPARK_TEMPLATE="projects/$PROJECT/locations/$REGION/notebookRuntimeTemplates/197505273697402880"
URL="https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$REGION/notebookExecutionJobs"
echo "SA=[$RUNNER_SA]"
echo "BUCKET=[$CODE_BUCKET]"
echo "MAIN=[$MAIN_TEMPLATE]"
echo "SPARK=[$SPARK_TEMPLATE]"
```

## 1. Single confirm POST — 07, with ack label (paste as one block)

```bash
TOKEN=$(gcloud auth print-access-token)
NB=$(base64 -w0 notebooks/07_scale_review.ipynb)
cat > /tmp/nbjob.json <<EOF
{"displayName":"diag-test","directNotebookSource":{"content":"$NB"},"notebookRuntimeTemplateResourceName":"$MAIN_TEMPLATE","gcsOutputUri":"gs://$CODE_BUCKET/notebooks","serviceAccount":"$RUNNER_SA","executionTimeout":"900s","labels":{"aiplatform.googleapis.com/notebook_runtime_out_of_org_warning":"ack"}}
EOF
curl -sS -w "\n---HTTP_STATUS:%{http_code}---\n" -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @/tmp/nbjob.json
```

Expect `HTTP_STATUS:200` and JSON with `"name": "...notebookExecutionJobs/..."`.
If so, that one job is already running — proceed to step 2 to fire the rest.

## 2. Fan out ALL 8 notebooks via curl (paste as one block)

Submits each notebook with the ack label. Only `01_spark_via_connect` uses the Spark
template; everything else uses main. Each prints its HTTP status + job name.

```bash
TOKEN=$(gcloud auth print-access-token)
RUN_LABEL="demo-$(date +%Y%m%d)"
submit_nb () {
  local name="$1" tmpl="$2"
  local nb; nb=$(base64 -w0 "notebooks/${name}.ipynb")
  cat > /tmp/nb_${name}.json <<EOF
{"displayName":"sf-demo-${name}-${RUN_LABEL}","directNotebookSource":{"content":"${nb}"},"notebookRuntimeTemplateResourceName":"${tmpl}","gcsOutputUri":"gs://${CODE_BUCKET}/notebooks","serviceAccount":"${RUNNER_SA}","executionTimeout":"5400s","labels":{"aiplatform.googleapis.com/notebook_runtime_out_of_org_warning":"ack"}}
EOF
  echo "=== ${name} ==="
  curl -sS -w " [HTTP:%{http_code}]\n" -X POST "$URL" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d @/tmp/nb_${name}.json | grep -Eo '"name": "[^"]+"| \[HTTP:[0-9]+\]'
}

submit_nb model_playground     "$MAIN_TEMPLATE"
submit_nb 02_bigquery_native   "$MAIN_TEMPLATE"
submit_nb 07_scale_review      "$MAIN_TEMPLATE"
submit_nb 03_combo_and_ensemble "$MAIN_TEMPLATE"
submit_nb 05_spark_naive       "$MAIN_TEMPLATE"
submit_nb 06_spark_multi       "$MAIN_TEMPLATE"
submit_nb 04_ray_on_vertex     "$MAIN_TEMPLATE"
submit_nb 01_spark_via_connect "$SPARK_TEMPLATE"

echo
echo "Watch them render:"
echo "https://console.cloud.google.com/vertex-ai/colab/execution-jobs?project=$PROJECT"
```

Each line should end `[HTTP:200]`. Open the Executions menu to watch them render and
open the finished notebooks with outputs.
