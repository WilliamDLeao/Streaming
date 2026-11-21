"""Calculo de indicadores transitorios, rejeicao a perturbacao e requisitos."""

from typing import Any
import control as ct
import numpy as np

from .config import GanhosControlador, ParametrosSistema, RequisitosDesempenho
from .models import (
    calcular_margens,
    criar_controlador_tf,
    serializar_raizes,
    sintetizar_malha,
    verificar_estabilidade,
)
from .simulator import ConfigSimulacao, ResultadoSimulacao, simular_sistema


def calcular_metricas_degrau(
    t: np.ndarray, x: np.ndarray, params: ParametrosSistema, estavel: bool = True
) -> dict[str, Any]:
    """Calcula tr (10-90%), ts (2%), Mp (%) e erro estacionario ao degrau."""
    if not estavel:
        return {"tr_s": None, "ts_s": None, "Mp_pct": None,
                "ess_ev": None, "x_max_ev": None, "q_min_ev": None}
    xr = params.x_ref
    i10, i90 = int(np.argmax(x >= 0.10 * xr)), int(np.argmax(x >= 0.90 * xr))
    tr = float(t[i90] - t[i10]) if (x[i90] >= 0.90 * xr) else None
    fora = np.where(np.abs(x - xr) > 0.02 * xr)[0]
    ts = float(t[min(int(fora[-1]) + 1, len(t) - 1)]) if len(fora) > 0 else 0.0
    x_max = float(np.max(x))
    return {
        "tr_s": round(tr, 4) if tr is not None else None,
        "ts_s": round(ts, 4),
        "Mp_pct": round(float(max(0.0, (x_max - xr) / xr * 100.0)), 4),
        "ess_ev": round(float(xr - x[-1]), 4),
        "x_max_ev": round(x_max, 4),
        "q_min_ev": round(float(params.q_0 - x_max), 4),
    }


def calcular_metricas_perturbacao(
    sim: ResultadoSimulacao, ganhos: GanhosControlador,
    params: ParametrosSistema, estavel: bool = True,
) -> dict[str, Any]:
    """Avalia erro permanente e recuperacao apos degrau de carga em t_dist."""
    if not estavel:
        return {"ess_dist_teorico_ev": None, "ess_dist_sim_ev": None,
                "x_min_apos_dist_ev": None, "q_max_apos_dist_ev": None,
                "ts_recup_2pct_s": None}
    ess_teo = 0.0 if ganhos.ki > 0.0 else round(params.dlambda_step / ganhos.kp, 4)
    mask = sim.t >= params.t_dist
    t_d, x_d, q_d = sim.t[mask], sim.x[mask], sim.q[mask]
    ess_sim = float(params.x_ref - x_d[-1])
    fora = np.where(np.abs(x_d - (params.x_ref - ess_sim)) > 0.02 * params.x_ref)[0]
    ts_rec = (float(t_d[min(int(fora[-1]) + 1, len(t_d) - 1)] - params.t_dist)
              if len(fora) > 0 else 0.0)
    return {
        "ess_dist_teorico_ev": ess_teo,
        "ess_dist_sim_ev": round(ess_sim, 4),
        "x_min_apos_dist_ev": round(float(np.min(x_d)), 4),
        "q_max_apos_dist_ev": round(float(np.max(q_d)), 4),
        "ts_recup_2pct_s": round(ts_rec, 4),
    }


def calcular_metricas_capacidade(
    sim: ResultadoSimulacao, params: ParametrosSistema
) -> dict[str, float]:
    """Calcula extremos de mu_c(t), mu(t) e tempo total em saturacao."""
    dt = float(sim.t[1] - sim.t[0])
    sat = (sim.u_cmd > params.u_max) | (sim.u_cmd < params.u_min)
    return {
        "mu_c_max_evs": round(float(np.max(sim.muc)), 2),
        "mu_c_min_evs": round(float(np.min(sim.muc)), 2),
        "mu_max_evs": round(float(np.max(sim.mu)), 2),
        "mu_min_evs": round(float(np.min(sim.mu)), 2),
        "tempo_saturado_s": round(float(np.sum(sat) * dt), 4),
    }


def avaliar_requisitos(
    met_step: dict[str, Any], ess_dist: float | None, pm_deg: float | None,
    mu_min: float, mu_max: float, estavel: bool, reqs: RequisitosDesempenho,
) -> dict[str, bool]:
    """Verifica atendimento individual e global aos criterios de projeto."""
    if not estavel:
        return {"a_tr": False, "b_ts": False, "c_Mp": False, "d_ess": False,
                "e_PM": False, "f_cap": False, "atende_todos": False}
    tr_ok = (met_step["tr_s"] is not None) and (met_step["tr_s"] <= reqs.tr_max_s)
    ts_ok = (met_step["ts_s"] is not None) and (met_step["ts_s"] <= reqs.ts_max_s)
    mp_ok = (met_step["Mp_pct"] is not None) and (met_step["Mp_pct"] <= reqs.mp_max_pct)
    ess_ok = (met_step["ess_ev"] is not None
              and abs(met_step["ess_ev"]) <= reqs.ess_max_ev
              and ess_dist is not None and abs(ess_dist) <= reqs.ess_max_ev)
    pm_ok = (pm_deg is not None) and (pm_deg >= reqs.pm_min_deg)
    cap_ok = (mu_min >= reqs.mu_min_evs - 1e-6) and (mu_max <= reqs.mu_max_evs + 1e-6)
    return {
        "a_tr": bool(tr_ok), "b_ts": bool(ts_ok), "c_Mp": bool(mp_ok),
        "d_ess": bool(ess_ok), "e_PM": bool(pm_ok), "f_cap": bool(cap_ok),
        "atende_todos": bool(
            tr_ok and ts_ok and mp_ok and ess_ok and pm_ok and cap_ok
        ),
    }


def analisar_configuracao(
    cfg: ConfigSimulacao, planta: ct.TransferFunction,
    params: ParametrosSistema, reqs: RequisitosDesempenho,
) -> tuple[dict[str, Any], ResultadoSimulacao, ResultadoSimulacao]:
    """Executa analise frequencial e simulacoes temporais (com/sem perturbacao)."""
    malha = sintetizar_malha(cfg.ganhos, planta)
    estavel, margens = malha["estavel"], malha["margens"]
    kw = dict(saturar=cfg.saturar, anti_windup=cfg.anti_windup,
              deriv_na_medicao=cfg.deriv_na_medicao, tt_aw=cfg.tt_aw)
    sim_step = simular_sistema(cfg.ganhos, params, incluir_perturbacao=False, **kw)
    sim_full = simular_sistema(cfg.ganhos, params, incluir_perturbacao=True, **kw)

    met_step = calcular_metricas_degrau(sim_step.t, sim_step.x, params, estavel)
    met_dist = calcular_metricas_perturbacao(sim_full, cfg.ganhos, params, estavel)
    cap = calcular_metricas_capacidade(sim_full, params)
    req_eval = avaliar_requisitos(
        met_step, met_dist["ess_dist_sim_ev"], margens["PM_deg"],
        cap["mu_min_evs"], cap["mu_max_evs"], estavel, reqs,
    )
    resumo: dict[str, Any] = {
        "nome": cfg.nome, "ganhos": cfg.ganhos.to_dict(),
        "saturacao": cfg.saturar, "anti_windup": cfg.anti_windup,
        "deriv_na_medicao": cfg.deriv_na_medicao, "estavel": estavel,
        "polos_cl": serializar_raizes(malha["polos_cl"]),
        "zeros_cl": serializar_raizes(malha["zeros_cl"]),
        "margens": margens, "metricas_degrau": met_step,
        "metricas_perturbacao": met_dist, "capacidade": cap,
        "requisitos": req_eval,
    }
    if cfg.incluir_tf_ideal and cfg.ganhos.kd > 0.0:
        c_id = criar_controlador_tf(cfg.ganhos, usar_filtro=False)
        t_id = ct.feedback(c_id * planta, 1)
        _, y_id = ct.step_response(t_id, sim_step.t)
        polos_id = ct.poles(t_id)
        resumo["ideal_sem_filtro"] = {
            "polos_cl": serializar_raizes(polos_id),
            "margens": calcular_margens(c_id * planta),
            "metricas_degrau": calcular_metricas_degrau(
                sim_step.t, params.x_ref * y_id, params,
                verificar_estabilidade(polos_id),
            ),
        }
    return resumo, sim_step, sim_full
