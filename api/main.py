from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os
import json


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT GENFORGE MODULES
# ============================================================

try:
    from orchestration.graph import run_genforge
    from optimizer.experiment_history import ExperimentHistory
except Exception as e:
    print(f"Import warning: {e}")
    run_genforge = None
    ExperimentHistory = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="GenForge AI API",
    description="Synthetic Data Generation and Optimization API",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    FRONTEND_URL
]

# Remove duplicates
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
        "name": "GenForge AI API",
        "version": "1.0.0",
        "status": "running",
        "message": "GenForge AI backend is running successfully."
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "GenForge AI API"
    }


# ============================================================
# BACKGROUND EXPERIMENT FUNCTION
# ============================================================

def run_experiment_background():
    try:
        print("Starting GenForge AI experiment...")

        if run_genforge is None:
            raise RuntimeError(
                "GenForge pipeline could not be imported."
            )

        result = run_genforge()

        print("GenForge AI experiment completed.")

        return result

    except Exception as e:
        print(f"GenForge experiment failed: {e}")
        raise


# ============================================================
# START EXPERIMENT
# ============================================================

@app.post("/experiment")
def start_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks
):
    try:

        background_tasks.add_task(
            run_experiment_background
        )

        return {
            "status": "started",
            "message": "GenForge AI experiment started in the background.",
            "config": {
                "max_iterations": request.max_iterations,
                "critic_threshold": request.critic_threshold,
                "development_class_5_limit": request.development_class_5_limit
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# GET ALL EXPERIMENTS
# ============================================================

@app.get("/experiments")
def get_experiments():

    try:

        if ExperimentHistory is None:
            raise RuntimeError(
                "ExperimentHistory could not be imported."
            )

        history = ExperimentHistory()

        if hasattr(history, "get_all_experiments"):
            experiments = history.get_all_experiments()

        elif hasattr(history, "get_history"):
            experiments = history.get_history()

        else:
            experiments = []

        return {
            "status": "success",
            "experiments": experiments
        }

    except Exception as e:

        return {
            "status": "error",
            "experiments": [],
            "message": str(e)
        }


# ============================================================
# GET SINGLE EXPERIMENT
# ============================================================

@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):

    try:

        if ExperimentHistory is None:
            raise RuntimeError(
                "ExperimentHistory could not be imported."
            )

        history = ExperimentHistory()

        if hasattr(history, "get_experiment"):
            experiment = history.get_experiment(
                experiment_id
            )

        elif hasattr(history, "get_by_id"):
            experiment = history.get_by_id(
                experiment_id
            )

        else:
            experiment = None

        if experiment is None:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found."
            )

        return {
            "status": "success",
            "experiment": experiment
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# GET BEST EXPERIMENT RESULT
# ============================================================

@app.get("/experiments/best/result")
def get_best_experiment_result():

    try:

        if ExperimentHistory is None:
            raise RuntimeError(
                "ExperimentHistory could not be imported."
            )

        history = ExperimentHistory()

        if hasattr(history, "get_best_experiment"):
            result = history.get_best_experiment()

        elif hasattr(history, "get_best"):
            result = history.get_best()

        else:
            result = None

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="No experiment results found."
            )

        return {
            "status": "success",
            "result": result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# COMPARE TWO EXPERIMENTS
# ============================================================

@app.get(
    "/experiments/compare/{experiment_id_1}/{experiment_id_2}"
)
def compare_experiments(
    experiment_id_1: str,
    experiment_id_2: str
):

    try:

        if ExperimentHistory is None:
            raise RuntimeError(
                "ExperimentHistory could not be imported."
            )

        history = ExperimentHistory()

        experiment_1 = None
        experiment_2 = None

        if hasattr(history, "get_experiment"):
            experiment_1 = history.get_experiment(
                experiment_id_1
            )

            experiment_2 = history.get_experiment(
                experiment_id_2
            )

        elif hasattr(history, "get_by_id"):
            experiment_1 = history.get_by_id(
                experiment_id_1
            )

            experiment_2 = history.get_by_id(
                experiment_id_2
            )

        if experiment_1 is None:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id_1} not found."
            )

        if experiment_2 is None:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id_2} not found."
            )

        return {
            "status": "success",
            "experiment_1": experiment_1,
            "experiment_2": experiment_2
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

@app.get("/system")
def system_info():

    try:

        import torch

        cuda_available = torch.cuda.is_available()

        model_paths = [
            os.path.join(
                PROJECT_ROOT,
                "models"
            ),
            os.path.join(
                PROJECT_ROOT,
                "evaluation"
            )
        ]

        checkpoints = []

        for path in model_paths:

            if os.path.exists(path):

                for root_dir, dirs, files in os.walk(path):

                    for file in files:

                        if file.endswith(
                            (".pth", ".pt", ".ckpt")
                        ):
                            checkpoints.append(
                                os.path.join(
                                    root_dir,
                                    file
                                )
                            )

        return {
            "status": "success",
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "gpu_count": (
                torch.cuda.device_count()
                if cuda_available
                else 0
            ),
            "checkpoints": checkpoints
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# IMBALANCED DATASET RESEARCH
# ============================================================

@app.get("/research/imbalanced")
def imbalanced_research():

    file_path = os.path.join(
        PROJECT_ROOT,
        "analysis",
        "imbalanced_analysis.json"
    )

    try:

        if not os.path.exists(file_path):
            return {
                "status": "not_found",
                "message": "Imbalanced analysis file not found."
            }

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return {
            "status": "success",
            "data": data
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# MULTI-SEED RESEARCH
# ============================================================

@app.get("/research/multiseed")
def multiseed_research():

    file_path = os.path.join(
        PROJECT_ROOT,
        "analysis",
        "multiseed_results.json"
    )

    try:

        if not os.path.exists(file_path):
            return {
                "status": "not_found",
                "message": "Multi-seed results file not found."
            }

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return {
            "status": "success",
            "data": data
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 60)
    print("GenForge AI API")
    print("Backend started successfully")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Frontend URL: {FRONTEND_URL}")
    print("=" * 60)