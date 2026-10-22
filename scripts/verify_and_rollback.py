import os
import subprocess
import sys
import time

import requests

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
KUBE_NAMESPACE = os.getenv("KUBE_NAMESPACE", "ml-service")
APP_NAME = os.getenv("APP_NAME", "ml-app")
ROLLBACK_MODE = os.getenv("ROLLBACK_MODE", "deployment")

WARMUP_SECONDS = int(os.getenv("WARMUP_SECONDS", "60"))
WINDOW = os.getenv("OBSERVATION_WINDOW", "5m")
MAX_5XX_RATE = float(os.getenv("MAX_5XX_RATE", "0.01"))
MAX_P95_LATENCY = float(os.getenv("MAX_P95_LATENCY", "1.0"))
MIN_RPS = float(os.getenv("MIN_RPS", "0.01"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"


def query_prometheus(query: str):
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                     params={"query": query}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data["status"] != "success" or not data["data"]["result"]:
        return None
    return float(data["data"]["result"][0]["value"][1])


def check_metrics():
    reasons, values = [], {}
    ns = KUBE_NAMESPACE

    q_5xx = (f'sum(rate(http_requests_total{{kubernetes_namespace="{ns}",'
             f'status=~"5.."}}[{WINDOW}])) / '
             f'sum(rate(http_requests_total{{kubernetes_namespace="{ns}"}}[{WINDOW}]))')
    try:
        v = query_prometheus(q_5xx); values["5xx_rate"] = v
        if v is not None and v > MAX_5XX_RATE:
            reasons.append(f"5xx {v:.4f} > {MAX_5XX_RATE}")
    except Exception as e:
        print(f"[verify] WARN 5xx: {e}"); values["5xx_rate"] = None

    q_p95 = (f'histogram_quantile(0.95, sum(rate('
             f'http_request_duration_seconds_bucket{{kubernetes_namespace="{ns}"}}'
             f'[{WINDOW}])) by (le))')
    try:
        v = query_prometheus(q_p95); values["p95"] = v
        if v is not None and v > MAX_P95_LATENCY:
            reasons.append(f"p95 {v:.3f}s > {MAX_P95_LATENCY}s")
    except Exception as e:
        print(f"[verify] WARN p95: {e}"); values["p95"] = None

    q_rps = f'sum(rate(http_requests_total{{kubernetes_namespace="{ns}"}}[{WINDOW}]))'
    try:
        v = query_prometheus(q_rps); values["rps"] = v
        if v is not None and v < MIN_RPS:
            reasons.append(f"rps {v:.4f} < {MIN_RPS}")
    except Exception as e:
        print(f"[verify] WARN rps: {e}"); values["rps"] = None

    return (len(reasons) == 0, reasons, values)


def rollback():
    if ROLLBACK_MODE == "canary":
        manifest = """apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-app
  namespace: ml-service
spec:
  hosts: [ml-app]
  http:
    - route:
        - destination: { host: ml-app, subset: stable }
          weight: 100
        - destination: { host: ml-app, subset: canary }
          weight: 0
"""
        if DRY_RUN:
            print("[verify] DRY_RUN canary rollback"); return
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=manifest, text=True, check=True,
        )
        print("[verify] Traffic shifted 100% → stable")
    else:
        cmd = ["kubectl", "rollout", "undo",
               f"deployment/{APP_NAME}", "-n", KUBE_NAMESPACE]
        if DRY_RUN:
            print(f"[verify] DRY_RUN: {' '.join(cmd)}"); return
        subprocess.run(cmd, check=True)


def main():
    print(f"[verify] mode={ROLLBACK_MODE}, warmup={WARMUP_SECONDS}s")
    time.sleep(WARMUP_SECONDS)

    try:
        ok, reasons, values = check_metrics()
    except Exception as e:
        print(f"[verify] Prometheus query failed: {e}"); sys.exit(2)

    print(f"[verify] values={values}")
    if ok:
        print("[verify] healthy"); sys.exit(0)

    print(f"[verify] FAIL: {reasons}")
    rollback()
    sys.exit(1)


if __name__ == "__main__":
    main()