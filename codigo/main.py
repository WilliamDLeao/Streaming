"""Orquestrador CLI das simulacoes e geracao de metricas e figuras."""

import json
import sys
from pathlib import Path

_DIR_CODIGO = Path(__file__).resolve().parent
if str(_DIR_CODIGO) not in sys.path:
    sys.path.insert(0, str(_DIR_CODIGO))

import numpy as np
from streaming_control import (
    DIR_FIGURAS,
    DIR_RESULTADOS,
    PARAMETROS_PADRAO,
    REQUISITOS_PADRAO,
    ConfigSimulacao,
    GanhosControlador,
    analisar_configuracao,
    criar_planta,
    gerar_todas_figuras,
)

Cenarios = tuple[
    list[float],
    dict[str, GanhosControlador],
    list[ConfigSimulacao],
    list[ConfigSimulacao],
]


def construir_cenarios(tau_f: float) -> Cenarios:
    """Define varredura proporcional e cenarios lineares e nao lineares."""
    k_vals = [0.08, 0.125, 0.369, 2.0]
    g_p = GanhosControlador(kp=0.369)
    g_pi = GanhosControlador(kp=0.35, ki=0.03)
    g_pi_inst = GanhosControlador(kp=0.50, ki=0.30)
    g_pd = GanhosControlador.alocar_pd(wn=0.80, zeta=0.80, tau_f=tau_f)
    g_pd_alt = GanhosControlador.alocar_pd(wn=1.00, zeta=0.85, tau_f=tau_f)
    g_pid_nom = GanhosControlador.alocar_pid(0.90, 1.0, 0.15, tau_f)
    g_pid_adj = GanhosControlador.alocar_pid(0.90, 1.0, 0.40, tau_f)

    ganhos_comp = {
        "P": g_p, "PI": g_pi, "PD": g_pd,
        "PID_nom": g_pid_nom, "PID_adj": g_pid_adj,
    }
    lin_cfgs = [
        ConfigSimulacao("P_ref", g_p, saturar=False),
        ConfigSimulacao("PI_estavel", g_pi, saturar=False),
        ConfigSimulacao("PI_instavel", g_pi_inst, saturar=False),
        ConfigSimulacao("PD_filt", g_pd, saturar=False, incluir_tf_ideal=True),
        ConfigSimulacao("PD_alt_filt", g_pd_alt, saturar=False),
        ConfigSimulacao("PID_nom_filt", g_pid_nom, saturar=False,
                        incluir_tf_ideal=True),
        ConfigSimulacao("PID_adj_filt", g_pid_adj, saturar=False,
                        incluir_tf_ideal=True),
    ]
    sat_cfgs = [
        ConfigSimulacao("P_ref_sat", g_p),
        ConfigSimulacao("PI_estavel_sat_noAW", g_pi, anti_windup="none"),
        ConfigSimulacao("PI_estavel_sat_clamp", g_pi, anti_windup="clamp"),
        ConfigSimulacao("PI_instavel_sat", g_pi_inst, anti_windup="none"),
        ConfigSimulacao("PD_filt_sat_Derr", g_pd, deriv_na_medicao=False),
        ConfigSimulacao("PD_filt_sat_Dmeas", g_pd, deriv_na_medicao=True),
        ConfigSimulacao("PID_nom_sat_noAW_Derr", g_pid_nom, anti_windup="none"),
        ConfigSimulacao("PID_nom_sat_clamp_Derr", g_pid_nom, anti_windup="clamp"),
        ConfigSimulacao("PID_nom_sat_clamp_Dmeas", g_pid_nom,
                        anti_windup="clamp", deriv_na_medicao=True),
        ConfigSimulacao("PID_nom_sat_backcalc_Dmeas", g_pid_nom,
                        anti_windup="backcalc", deriv_na_medicao=True),
        ConfigSimulacao("PID_adj_sat_clamp_Dmeas", g_pid_adj,
                        anti_windup="clamp", deriv_na_medicao=True),
        ConfigSimulacao("PID_adj_sat_backcalc_Dmeas", g_pid_adj,
                        anti_windup="backcalc", deriv_na_medicao=True),
    ]
    return k_vals, ganhos_comp, lin_cfgs, sat_cfgs


def main() -> None:
    """Executa simulacoes, gera figuras e exporta o JSON de metricas."""
    params, reqs = PARAMETROS_PADRAO, REQUISITOS_PADRAO
    planta = criar_planta(params)
    k_vals, ganhos_comp, lin_cfgs, sat_cfgs = construir_cenarios(params.tau_f)

    res_p_sweep = {}
    for k in k_vals:
        cfg_k = ConfigSimulacao(f"P (K={k})", GanhosControlador(kp=k),
                                saturar=False)
        res_k, _, _ = analisar_configuracao(cfg_k, planta, params, reqs)
        res_k["wn_rads"] = round(float(np.sqrt(k / 2.0)), 4)
        res_k["zeta"] = round(float(1.0 / np.sqrt(8.0 * k)), 4)
        res_p_sweep[f"K_{k}"] = res_k

    resultados_lin, sims_lin = {}, {}
    for cfg in lin_cfgs:
        res, _, s_full = analisar_configuracao(cfg, planta, params, reqs)
        resultados_lin[cfg.nome] = res
        sims_lin[cfg.nome] = s_full

    resultados_sat, sims_sat = {}, {}
    for cfg in sat_cfgs:
        res, _, s_full = analisar_configuracao(cfg, planta, params, reqs)
        resultados_sat[cfg.nome] = res
        sims_sat[cfg.nome] = s_full

    dados_ma = gerar_todas_figuras(
        k_vals, ganhos_comp, sims_lin, sims_sat, planta, params, DIR_FIGURAS
    )
    saida = {
        "parametros_sistema": params.to_dict(reqs),
        "malha_aberta": {"polos": [0.0, -0.5], "zeros": [], "passos": dados_ma},
        "proporcional_sweep": res_p_sweep,
        "lineares": resultados_lin,
        "nao_lineares_saturacao": resultados_sat,
    }
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    caminho_json = DIR_RESULTADOS / "resultados_metricas.json"
    with caminho_json.open("w", encoding="utf-8") as arq:
        json.dump(saida, arq, indent=2, ensure_ascii=False)
    print(f"Simulacoes concluidas: {caminho_json} e 16 figuras em {DIR_FIGURAS}.")


if __name__ == "__main__":
    main()
