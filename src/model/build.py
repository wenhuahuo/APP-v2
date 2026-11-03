"""按配置的 `model_type` 构造模型：APP 主线与各基线共用同一套训练、评测入口。"""

from app_v1 import APPSolverV1Baseline
from appsolver_gnot import APPSolverGNOT
from appsolver_oformer import APPSolverOFormer
from appsolver_transolver import APPSolverTransolver
from appsolver_upt import APPSolverUPT
from appsolver_v2 import APPSolverV2
from coral import CORALBaseline
from gino import GINOBaseline
from gnot import GNOTBaseline
from oformer import OFormerBaseline
from transolver import TransolverBaseline
from upt import UPTBaseline

MODELS = {
    "appsolver_v2": APPSolverV2,
    "appsolver_oformer": APPSolverOFormer,
    "appsolver_transolver": APPSolverTransolver,
    "appsolver_upt": APPSolverUPT,
    "appsolver_gnot": APPSolverGNOT,
    "appsolver_v1": APPSolverV1Baseline,
    "gnot": GNOTBaseline,
    "upt": UPTBaseline,
    "transolver": TransolverBaseline,
    "oformer": OFormerBaseline,
    "gino": GINOBaseline,
    "coral": CORALBaseline,
}


def build_model(config):
    return MODELS[config["model_type"]](**config["model"])
