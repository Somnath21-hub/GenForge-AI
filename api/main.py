
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import sys
import os
import json


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ============================================================
# GENFORGE IMPORTS
# ============================================================

from orchestration.graph import run_genforge
from optimizer.experiment_history import ExperimentHistory


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="GenForge AI API",
    description=(
        "Backend API for autonomous synthetic data "
        "generation and ML experiment optimization."
    ),
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# Local development:
#   FRONTEND_URL is not required.
#
# Production:
#   Set FRONTEND_URL in Render to your Vercel URL.
#
# Example:
#   FRONTEND_URL=https://genforge-ai.vercel.app

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    FRONTEND_URL
]

# Remove duplicate origins
ALLOWED_ORIGINS = list(set(ALLOWED_ORIGINS))


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ExperimentRequest(BaseModel):

    max_iterations: Optional[int] = 3

    critic_threshold: Optional[float] = 0.90

    development_class_5_limit: Optional[int] = 1000


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "GenForge AI",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# BACKGROUND EXPERIMENT
# ============================================================

def run_experiment_background():

    try:

        print("\n========================================")
        print("GENFORGE EXPERIMENT STARTED")
        print("========================================\n")

        run_genforge()

        print("\n========================================")
        print("GENFORGE EXPERIMENT COMPLETED")
        print("========================================\n")

    except Exception as error:

        print("\n========================================")
        print("GENFORGE EXPERIMENT FAILED")
        print("========================================")

        print("Error:", error)

        print("========================================\n")


# ============================================================
# START EXPERIMENT
# ============================================================

@app.post("/experiment")
def start_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks
):

    background_tasks.add_task(
        run_experiment_background
    )

    return {

        "status": "started",

        "message": (
            "GenForge experiment started "
            "in the background."
        ),

        "config": {

            "max_iterations":
                request.max_iterations,

            "critic_threshold":
                request.critic_threshold,

            "development_class_5_limit":
                request.development_class_5_limit
        }
    }


# ============================================================
# GET ALL EXPERIMENTS
# ============================================================

@app.get("/experiments")
def get_experiments():

    try:

        history = ExperimentHistory()

        experiments = history.get_history()

        return {

            "total_experiments":
                len(experiments),

            "experiments":
                experiments
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load experiments: {str(error)}"
        )


# ============================================================
# GET SINGLE EXPERIMENT
# ============================================================

@app.get("/experiments/{experiment_id}")
def get_experiment(
    experiment_id: int
):

    try:

        history = ExperimentHistory()

        experiment = history.get_experiment(
            experiment_id
        )

        if experiment is None:

            raise HTTPException(
                status_code=404,
                detail="Experiment not found"
            )

        return experiment

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load experiment: {str(error)}"
        )


# ============================================================
# GET BEST EXPERIMENT
# ============================================================

@app.get("/experiments/best/result")
def get_best_experiment():

    try:

        history = ExperimentHistory()

        best = history.get_best_experiment()

        if best is None:

            return {
                "message": "No experiments available."
            }

        return best

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load best experiment: {str(error)}"
        )


# ============================================================
# COMPARE TWO EXPERIMENTS
# ============================================================

@app.get(
    "/experiments/compare/"
    "{experiment_id_1}/"
    "{experiment_id_2}"
)
def compare_experiments(
    experiment_id_1: int,
    experiment_id_2: int
):

    try:

        history = ExperimentHistory()

        comparison = history.compare_experiments(
            experiment_id_1,
            experiment_id_2
        )

        if comparison is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "One or both experiments "
                    "were not found."
                )
            )

        return comparison

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to compare experiments: "
                f"{str(error)}"
            )
        )


# ============================================================
# SYSTEM STATUS & HARDWARE
# ============================================================

@app.get("/system")
def get_system_status():

    try:

        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:

            device_name = torch.cuda.get_device_name(0)

            device_count = torch.cuda.device_count()

        else:

            device_name = "CPU (Fallback)"

            device_count = 0


        checkpoints = {

            "cvae_v1":
                os.path.exists(
                    os.path.join(
                        PROJECT_ROOT,
                        "models",
                        "conditional_vae.pth"
                    )
                ),

            "cvae_v2":
                os.path.exists(
                    os.path.join(
                        PROJECT_ROOT,
                        "models",
                        "conditional_vae_v2.pth"
                    )
                ),

            "cvae_v3":
                os.path.exists(
                    os.path.join(
                        PROJECT_ROOT,
                        "models",
                        "conditional_vae_v3.pth"
                    )
                ),

            "baseline_cnn":
                os.path.exists(
                    os.path.join(
                        PROJECT_ROOT,
                        "evaluation",
                        "baseline_cnn.pth"
                    )
                ),

            "independent_evaluator":
                os.path.exists(
                    os.path.join(
                        PROJECT_ROOT,
                        "models",
                        "independent_evaluator.pth"
                    )
                )
        }


        return {

            "status": "online",

            "version": "1.0.0",

            "framework":
                "FastAPI + Uvicorn",

            "pytorch_version":
                torch.__version__,

            "cuda_available":
                cuda_available,

            "gpu_name":
                device_name,

            "device_count":
                device_count,

            "langgraph_loaded":
                True,

            "checkpoints":
                checkpoints
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"System check failed: {str(error)}"
        )


# ============================================================
# RESEARCH - IMBALANCED DATA
# ============================================================

@app.get("/research/imbalanced")
def get_imbalanced_research():

    path = os.path.join(
        PROJECT_ROOT,
        "experiments",
        "imbalanced_severe_results.json"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Imbalanced results not found"
        )

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to read research data: {str(error)}"
        )


# ============================================================
# RESEARCH - MULTI SEED
# ============================================================

@app.get("/research/multiseed")
def get_multiseed_research():

    path = os.path.join(
        PROJECT_ROOT,
        "experiments",
        "multi_seed_results.json"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Multi-seed results not found"
        )

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to read research data: {str(error)}"
        )
