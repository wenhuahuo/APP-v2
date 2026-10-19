"""按配置的 `model_type` 构造模型：APP 主线与四个基线共用同一套训练、评测入口。"""

from app_v1 import APPSolverV1Baseline
from appsolver_v2 import APPSolverV2
from gnot import GNOTBaseline
from oformer import OFormerBaseline
from transolver import TransolverBaseline
from upt import UPTBaseline

MODELS = {
    "appsolver_v2": APPSolverV2,
    "appsolver_v1": APPSolverV1Baseline,
    "gnot": GNOTBaseline,
    "upt": UPTBaseline,
    "transolver": TransolverBaseline,
    "oformer": OFormerBaseline,
}


def build_model(config):
    return MODELS[config["model_type"]](**config["model"])
