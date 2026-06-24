# TLDR

Renaming service from `completion.kalygo.io` -> `agent.kalygo.io`

"Completion API" → **"Agent API"**. Identity changes across code, infra, and DNS:

| Thing | Old | New |
|---|---|---|
| Branding / app title | Kalygo3 Completion API | **Kalygo3 Agent API** |
| Python dist name | `kalygo3-completion-api` | `kalygo3-agent-api` |
| DB `application_name` | `kalygo3-completion` | `kalygo3-agent` |
| Cloud Run service | `kalygo-completion-api-service` | `kalygo-agent-api-service` |
| Artifact Registry repo / image | `kalygo-completion-api` | `kalygo-agent-api` |
| Public domain | `completion.kalygo.io` | `agent.kalygo.io` |
| Route 53 record name | `completion` | `agent` |

GCP project `kalygo-436411` · Cloud Run region `us-east1` · Artifact Registry `us-central1`.

---

## 1. In-repo changes — DONE ✅

Already committed to this repo (no action needed):

- App title/description/docstrings, `README.md`, `pyproject.toml` name, `logging_config.py`, `src/__init__.py`, `alembic/env.py`, mirror-guard test docstrings.
- `src/db/database.py` → `application_name: "kalygo3-agent"`.
- `service.yaml` → `metadata.name: kalygo-agent-api-service` + new image path.
- `.github/workflows/cicd.yaml` → `SERVICE_NAME` + `ARTIFACTORY_URL` (also fixed the stale test `CREDENTIALS_ENCRYPTION_KEY`).
- `scripts/load_test_*` + DNS doc → `agent.kalygo.io`.

> Generic "completion" wording (e.g. "streaming LLM completion", the `agent_completion` handler) is left as-is — it describes the *act*, not the service.

---

## 2. Manual GCP / DNS cutover — TODO 🔴

These cannot be done from the repo. Do them in order. **Run old + new in parallel and only decommission the old once clients are migrated** — this keeps `completion.kalygo.io` serving the whole time.

### 2a. Create the new Artifact Registry repo (else `docker push` fails)
```bash
gcloud artifacts repositories create kalygo-agent-api \
  --repository-format=docker \
  --location=us-central1 \
  --project=kalygo-436411
```

### 2b. Deploy → creates the new Cloud Run service
Push to `main` (CI builds the image into `kalygo-agent-api` and deploys
`kalygo-agent-api-service`), or do it by hand:
```bash
REPO=us-central1-docker.pkg.dev/kalygo-436411/kalygo-agent-api/kalygo-agent-api
TAG=manual-$(date +%s)
docker build -f Dockerfile.prod -t "$REPO:$TAG" .
docker push "$REPO:$TAG"
# service.yaml references the image untagged; pin this tag before replace
# (a fresh repo has no :latest — see the same step in cicd.yaml).
sed -i "s|image: $REPO\$|image: $REPO:$TAG|" service.yaml
gcloud run services replace service.yaml --region us-east1 --project kalygo-436411
git checkout service.yaml   # discard the local pin
```
The old `kalygo-completion-api-service` keeps running and serving
`completion.kalygo.io` — untouched.

### 2c. Map the new domain to the new service
```bash
gcloud beta run domain-mappings create \
  --service kalygo-agent-api-service \
  --domain agent.kalygo.io \
  --region us-east1 --project kalygo-436411
```
This prints a CNAME/A record target — note it for the next step.

### 2d. Route 53 — add the DNS record
In the hosted zone for `kalygo.io`, add the record from 2c:
- **Name:** `agent`
- **Target:** the `ghs.googlehosted.com` (or A/AAAA set) returned by the domain mapping.

(See `READMEs/configuring-dns-for-gcr-microservice/` — same procedure that set up `completion`.)

### 2e. Wait for the managed TLS cert
`agent.kalygo.io` won't serve HTTPS until GCP provisions the cert (minutes–hours).
```bash
gcloud beta run domain-mappings describe \
  --domain agent.kalygo.io --region us-east1 --project kalygo-436411
# watch for CertificateProvisioned / Ready = True
```

### 2f. Verify
```bash
curl -v https://agent-api.kalygo.io/            # expect {"status":"OK!"}
# app title should read "Kalygo3 Agent API":
curl -s https://agent.kalygo.io/openapi.json | python -m json.tool | head
```

### 2g. Migrate clients
Point the frontend (and anything else hitting `completion.kalygo.io`) at
`agent.kalygo.io`. Confirm traffic has moved (Cloud Run metrics on the old service → ~0).

---

## 3. Decommission the old resources (only after 2g)

```bash
# Old domain mapping + DNS record
gcloud beta run domain-mappings delete --domain completion.kalygo.io \
  --region us-east1 --project kalygo-436411
#   …and delete the `completion` record in Route 53.

# Old Cloud Run service
gcloud run services delete kalygo-completion-api-service \
  --region us-east1 --project kalygo-436411

# Old image repo (optional — keeps old images for rollback if left)
gcloud artifacts repositories delete kalygo-completion-api \
  --location=us-central1 --project=kalygo-436411
```

## 4. Rollback

Until step 3, rollback is trivial: clients still work on `completion.kalygo.io`
(old service untouched). Just point clients back and pause the new deploy.

## Notes / gotchas
- **DB `application_name`** changes how this service shows up in `pg_stat_activity`
  and any monitoring dashboards keyed on `kalygo3-completion` — update those.
- The Postgres database itself is **shared** and unchanged; the rename touches no
  tables (the `completion-api mirror` models just got their comments reworded).
- IAM invoker bindings (`--allow-unauthenticated` is in CI), Cloud Run env-var
  secrets, and the `GCP_SA_KEY` secret are unaffected — they're not name-keyed.
