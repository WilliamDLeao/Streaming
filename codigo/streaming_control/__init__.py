"""Pacote modular para modelagem, simulacao e controle de filas em streaming."""

from .config import (
    CORES,
    DIR_FIGURAS,
    DIR_RESULTADOS,
    PARAMETROS_PADRAO,
    REQUISITOS_PADRAO,
    GanhosControlador,
    ParametrosSistema,
    RequisitosDesempenho,
)
from .evaluation import analisar_configuracao
from .models import criar_controlador_tf, criar_planta, sintetizar_malha
from .plotting import gerar_todas_figuras
from .simulator import (
    ConfigSimulacao,
    ResultadoSimulacao,
    simular_malha_aberta,
    simular_sistema,
)

__all__ = [
    "CORES",
    "DIR_FIGURAS",
    "DIR_RESULTADOS",
    "PARAMETROS_PADRAO",
    "REQUISITOS_PADRAO",
    "ConfigSimulacao",
    "GanhosControlador",
    "ParametrosSistema",
    "RequisitosDesempenho",
    "ResultadoSimulacao",
    "analisar_configuracao",
    "criar_controlador_tf",
    "criar_planta",
    "gerar_todas_figuras",
    "simular_malha_aberta",
    "simular_sistema",
    "sintetizar_malha",
]
