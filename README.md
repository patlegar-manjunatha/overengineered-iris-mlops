
#  Project : Enterprise MLOps Framework

### *Bridging the gap between Jupyter Notebooks and Production-Scale AI.*

##  The Mission

In modern AI engineering, **the model is the easy part.** The challenge lies in the 95% of code surrounding it: configuration, tracking, serving, and infrastructure.

**Project IRIS** uses the Iris dataset as a stable baseline to demonstrate a **Production-Grade MLOps Ecosystem**. By keeping the model simple, we prioritize engineering maturity: service decoupling, headless authentication, and automated artifact lifecycles.

## System Architecture & Design Decisions

Unlike monolithic ML scripts, IRIS is architected as a distributed system.

### 1. Decoupled Microservices

* **Prediction Engine (The Serving Layer):** A FastAPI-based service designed for high availability. It implements **Dynamic Artifact Loading**, ensuring zero-downtime when a new model is trained.
* **Training Engine (The Computation Layer):** An isolated worker service that handles heavy lifting (GridSearch, Cross-Validation). By separating this, we can scale compute resources for training independently from the prediction API.

### 2. The Artifact Handshake

Implemented a **Volume-Shared Persistence** pattern. When the Training Engine finishes, it serializes the `StandardScaler` and `LogisticRegression` objects into a shared volume, triggering a signal to the Prediction Engine to reload the state — simulating a real-world Model Registry flow.

 

## Governance & Reliability

###  Enterprise Tracking (MLflow + DagsHub)

Every training run is an audited event. We don't just log accuracy; we log the entire hyperparameter space:

* **Hyperparameter Audit:**  vs  penalties,  coefficients, and ElasticNet ratios.
* **Cloud Observability:** Remote tracking via DagsHub ensures the "Brain" of the project is visible to the whole team, not just trapped on a local machine.

### "Shift-Left" Testing & Mocking

Production systems fail at the boundaries. Our testing strategy focuses on:

* **Boundary Testing:** Mocks the HTTP handshake between services using `httpx`.
* **Headless Validation:** Ensures the system can authenticate and log metrics in a "No-Browser" environment (Crucial for CI/CD and Kubernetes).
* **Deterministic Coverage:** Achieving high test coverage to ensure the logic of the pipeline is bulletproof before deployment.

 

## Engineering Blueprint

```bash
├── .github/workflows/  # CI/CD: Automated testing and image builds
├── prediction/         # High-availability Inference API
│   ├── main.py         # Logic for Dynamic Model Reloading
│   └── Dockerfile      # Multi-stage optimized build
├── training/           # On-demand Training Worker
│   ├── train.py        # GridSearch + MLflow Governance logic
│   └── Dockerfile      # Headless environment configuration
├── k8s/                # Kubernetes Manifests (Deployments, Services, PVC)
└── docker-compose.yaml # Local Environment Orchestration

```

 

## Scalability & Future Evolution

IRIS is built to grow. The current architecture allows for:

1. **Kubernetes Migration:** Already includes manifests for **HPA (Horizontal Pod Autoscaling)** to handle traffic spikes.
2. **Model Drift Detection:** Designed to integrate with Prometheus for monitoring feature distribution changes.
3. **A/B Testing:** The decoupled nature allows for "Blue-Green" model deployments without interrupting service.

 

## The Bottom Line

**This isn't a data science demo; it's a systems engineering showcase.** It proves the ability to build machine learning systems that are **Observable**, **Testable**, and **Deployable**. While the data is Iris, the architecture is **Enterprise**.

 

## Quickstart (Simulate the Cloud Locally)

1. Clone Repo
 ```bash
   git clone https://github.com/patlegar-manjunatha/overengineered-iris-mlops.git
 ```
2. RUN  
```bash
docker-compose up --build
```
3. Navigate to `http://localhost:8000` to trigger your first managed training run.

