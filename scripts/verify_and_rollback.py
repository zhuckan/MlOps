import os
import sys
import time
import subprocess
import requests

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
KUBE_NAMESPACE = os.getenv("KUBE_NAMESPACE", "ml-service")
APP_NAME = os.getenv("APP_NAME", "ml-app")

WARMUP_SECONDS = int(os.getenv("WARMUP_SECONDS", "60"))
OBSERVATION_WINDOW = os.getenv("OBSERVATION_WINDOW", "5m")
MAX_5XX_RATE = float(os.getenv("MAX_5XX_RATE", "0.01"))
MAX_P95_LATENCY = float(os.getenv("MAX_P95_LATENCY", "1.0"))
MIN_RPS = float(os.getenv("MIN_RPS", "0.01"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"


def query_prometheus(query):
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    if data["status"] != "success":
        raise RuntimeError(f"PromQL failed: {data}")
    results = data["data"]["result"]
    if not results:
        return None
    return float(results[0]["value"][1])


def check_metrics():
    reasons = []
    values = {}

    ns = KUBE_NAMESPACE

    q_5xx = (
        f'sum(rate(http_requests_total{{kubernetes_namespace="{ns}",'
        f'status=~"5.."}}[{OBSERVATION_WINDOW}])) / '
        f'sum(rate(http_requests_total{{kubernetes_namespace="{ns}"}}[{OBSERVATION_WINDOW}]))'
    )
    try:
        rate_5xx = query_prometheus(q_5xx)
        values["5xx_rate"] = rate_5xx
        if rate_5xx is not None and rate_5xx > MAX_5XX_RATE:
            reasons.append(f"5xx rate {rate_5xx:.4f} > {MAX_5XX_RATE}")
    except Exception as e:
        print(f"[verify] WARN 5xx query failed: {e}")
        values["5xx_rate"] = None

    q_p95 = (
        f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket'
        f'{{kubernetes_namespace="{ns}"}}[{OBSERVATION_WINDOW}])) by (le))'
    )
    try:
        p95 = query_prometheus(q_p95)
        values["p95_latency"] = p95
        if p95 is not None and p95 > MAX_P95_LATENCY:
            reasons.append(f"p95 latency {p95:.3f}s > {MAX_P95_LATENCY}s")
    except Exception as e:
        print(f"[verify] WARN p95 query failed: {e}")
        values["p95_latency"] = None

    q_rps = (
        f'sum(rate(http_requests_total{{kubernetes_namespace="{ns}"}}[{OBSERVATION_WINDOW}]))'
    )
    try:
        rps = query_prometheus(q_rps)
        values["rps"] = rps
        if rps is not None and rps < MIN_RPS:
            reasons.append(f"rps {rps:.4f} < {MIN_RPS}")
    except Exception as e:
        print(f"[verify] WARN rps query failed: {e}")
        values["rps"] = None

    return (len(reasons) == 0, reasons, values)


def rollout_undo():
    cmd = ["kubectl", "rollout", "undo", f"deployment/{APP_NAME}", "-n", KUBE_NAMESPACE]
    if DRY_RUN:
        print(f"[verify] DRY_RUN, would run: {' '.join(cmd)}")
        return
    print(f"[verify] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    print(f"[verify] Prometheus: {PROMETHEUS_URL}")
    print(f"[verify] Warmup {WARMUP_SECONDS}s...")
    time.sleep(WARMUP_SECONDS)

    try:
        ok, reasons, values = check_metrics()
    except Exception as e:
        print(f"[verify] Cannot query Prometheus: {e}")
        sys.exit(2)

    print(f"[verify] Metrics: {values}")

    if ok:
        print("[verify] All metrics healthy")
        sys.exit(0)

    print(f"[verify] Failures: {reasons}")
    rollout_undo()
    print("[verify] Rollback triggered")
    sys.exit(1)


if __name__ == "__main__":
    main()