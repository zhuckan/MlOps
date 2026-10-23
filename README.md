# MLOps Pipeline: Credit Card Fraud Detection with Continuous Training

End-to-end MLOps pipeline that monitors a production fraud detection model, detects data and concept drift, automatically retrains when thresholds are breached, and safely rolls out new models through Istio Canary deployment and human approval.

## Key Features

### Continuous Training (CT)
- Airflow DAG with scheduled drift checks
- Retraining triggered automatically when drift exceeds thresholds
- Runs as a `KubernetesPodOperator` in an isolated K8s pod
- Champion/Challenger evaluation on a frozen holdout snapshot stored in MLflow

### Drift Monitoring
- **Data drift**: PSI + Kolmogorov-Smirnov test with Bonferroni correction
- **Concept drift**: prediction distribution, error rate on delayed labels
- Rolling window over partitioned traffic logs
- Versioned baseline tied to production model (MLflow `run_id`)
- HTML reports uploaded to S3-compatible storage (MinIO)

### Feature Store
- Feast with point-in-time correct joins
- Separate FeatureViews for features and delayed labels (no target leakage)
- Temporal split for training (no future data leakage)

### Safe Rollout (Istio Service Mesh)
- **Human-in-the-loop** approval via Airflow Variables with full audit trail
- **Shadow Deployment**: new model receives 100% mirrored traffic without affecting clients (using Istio `VirtualService` mirroring)
- **Canary Deployment**: gradual traffic shifting (90/10 → 50/50 → 100/0) via Istio weights
- Automated rollback on SLO violation (5xx rate, p95 latency, RPS) by shifting traffic back to stable

### CI/CD Integration (Airflow -> GitLab)
- When a new model is promoted, Airflow automatically triggers a GitLab CI/CD pipeline via REST API.
- GitLab pipeline builds the Docker images (Kaniko), pushes to a local Registry, and applies K8s manifests.
- Post-deploy Prometheus checks validate the new model before final promotion.

### Observability
- Prometheus scrapes business metrics (RPS, latency, errors, shadow agreement)
- Grafana dashboards for service health and model behavior
- Telegram notifications at every stage: drift, retrain, approval, deploy

### Infrastructure
- Kubernetes deployment with RBAC, ResourceQuota, LimitRange
- Liveness and readiness probes on all services
- Local Docker Registry + Minikube `insecure-registry` configuration
- Cross-network deployment via Tailscale (GitLab on remote host, K8s on local machine)

## Tech Stack

`Airflow` · `MLflow` · `Feast` · `Kubernetes` · `Minikube` · `Istio` · `Prometheus` · `Grafana` · `FastAPI` · `XGBoost` · `MinIO` · `GitLab CI/CD` · `Kaniko` · `Docker` · `Python` · `Tailscale`

## How It Works

1. **Drift Check** — Airflow DAG reads recent traffic from MinIO and compares against the active baseline using PSI and KS tests.
2. **Alert on Drift** — Prometheus metrics updated, Telegram notification sent with HTML report link.
3. **Automatic Retraining** — K8s Job pulls data via Feast point-in-time joins and trains a challenger model, logged to MLflow.
4. **Champion/Challenger** — Both models evaluated on the same frozen holdout for a fair comparison.
5. **Approval** — Operator approves via Airflow Variables; a full audit trail is recorded.
6. **Promote** — New model tagged as `Production`; baseline refreshed automatically.
7. **Trigger Deploy** — Airflow sends a REST API request to GitLab to start the deployment pipeline.
8. **Build & Deploy** — GitLab CI builds the images, pushes them, and deploys the shadow (canary) environment.
9. **Canary Rollout** — Operator manually approves traffic shifting (10% → 50% → 100%) in GitLab, monitored by Prometheus checks.
10. **Verify** — Post-deploy Prometheus checks validate 5xx rate, p95 latency, and RPS. If metrics drop, traffic is instantly reverted to the stable version.

## Project Structure

```text
.
├── dags/                    # Airflow DAGs (ml_monitoring_dag.py)
│   └── scripts/             # Drift monitor script
├── feature_repo/            # Feast definitions and data
├── manifests/               # Kubernetes manifests
│   ├── istio/               # Gateway, VirtualServices (Canary/Shadow), DestinationRule
│   └── monitoring/          # Prometheus, Grafana, Pushgateway
├── scripts/                 # verify_and_rollback.py
├── tests/                   # Pytest unit tests
├── app.py                   # FastAPI inference service
├── retrain.py               # Training pipeline (XGBoost + MLflow)
├── entrypoint.sh            # Model downloader for stable pod
├── download_model_for_shadow.py # Model downloader for canary image
├── simulate_traffic.py      # Drift simulation script
├── .gitlab-ci.yml           # CI/CD pipeline definition
└── Dockerfile*              # Multi-stage Dockerfiles for API, Canary, Retrain, Drift
```

## Known Limitations (for educational purposes)
- The current `retrain.py` uses a static `features.parquet` file for training instead of fetching live data from MinIO, which results in identical metric scores for Challenger and Champion models. In a true production environment, the training script should trigger `feast materialize` and read the latest partitioned data.
- The Drift Simulation script is decoupled from the Feature Store update process.

## Results
- Continuous training loop running end-to-end
- Frozen holdout ensuring fair champion/challenger comparison
- Drift detection with statistically sound thresholds
- Safe rollout with Istio Canary deployment, shadow scoring, and automated rollback
- Full observability through Prometheus and Grafana
- Human approval workflow with complete audit trail
- Automated handoff from MLOps (Airflow) to DevOps (GitLab CI/CD)