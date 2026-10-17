# MLOps Pipeline: Credit Card Fraud Detection with Continuous Training

End-to-end MLOps pipeline that monitors a production fraud detection model, detects data and concept drift, automatically retrains when thresholds are breached, and safely rolls out new models through shadow deployment and human approval.

## Key Features

### Continuous Training
- Airflow DAG with hourly drift checks
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

### Safe Rollout
- Human-in-the-loop approval via Airflow Variables with full audit trail
- **Istio Canary Deployment:** traffic shifting (90/10 → 50/50 → 100/0) via VirtualService weights
- **Shadow Deployment:** new model receives mirrored traffic without affecting clients
- Prediction agreement + probability difference metrics compared in Prometheus
- Automated rollback via `kubectl rollout undo` or Istio traffic shift on SLO violation (5xx rate, p95 latency, RPS)

### Observability
- Prometheus scrapes business metrics (RPS, latency, errors, shadow agreement)
- Grafana dashboards for service health and model behavior
- Telegram notifications at every stage: drift, retrain, approval

### Infrastructure
- Kubernetes deployment with RBAC, ResourceQuota, LimitRange
- Liveness and readiness probes on all services
- GitLab CI/CD pipeline with test, build, deploy and verify stages

## Tech Stack

`Airflow` · `MLflow` · `Feast` · `Kubernetes` · `Minikube` · `Prometheus` · `Grafana` · `FastAPI` · `XGBoost` · `MinIO` · `GitLab CI/CD` · `Docker` · `Python`

## How It Works

1. **Hourly drift check** — Airflow DAG reads recent traffic from MinIO and compares against the active baseline using PSI and KS tests.
2. **Alert on drift** — Prometheus metrics updated, Telegram notification sent with HTML report link.
3. **Automatic retraining** — K8s Job pulls fresh data via Feast point-in-time joins and trains a challenger model, logged to MLflow.
4. **Champion/Challenger** — Both models evaluated on the same frozen holdout for a fair comparison.
5. **Approval** — Operator approves via Airflow Variables; a full audit trail is recorded.
6. **Promote** — New model tagged as `Production`; baseline refreshed automatically.
7. **Deploy** — GitLab pipeline rebuilds the image and updates the K8s deployment.
8. **Verify** — Post-deploy Prometheus checks validate 5xx rate, p95 latency and RPS; automatic rollback on failure.

## Project Structure

```
.
├── dags/                    # Airflow DAGs
│   └── scripts/             # Drift monitor
├── feature_repo/            # Feast definitions
├── manifests/               # Kubernetes manifests
│   ├── istio/               # Gateway, VirtualService (Canary/Shadow), DestinationRule
│   └── monitoring/          # Prometheus, Grafana, Pushgateway
├── scripts/                 # verify_and_rollback
├── retrain.py               # Training pipeline
├── save_baseline.py         # Versioned baseline creator
├── switch_baseline.py       # Rollback helper
├── app.py                   # FastAPI inference service
├── simulate_traffic.py      # Drift simulation
└── .gitlab-ci.yml           # CI/CD pipeline
```

## Results

- Continuous training loop running end-to-end
- Frozen holdout ensuring fair champion/challenger comparison
- Drift detection with statistically sound thresholds
- Safe rollout with shadow scoring and automated rollback
- Full observability through Prometheus and Grafana
- Human approval workflow with complete audit trail
