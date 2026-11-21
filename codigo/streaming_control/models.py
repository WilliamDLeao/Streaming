"""Modelagem em funcao de transferencia, sintese e margens frequenciais."""

from typing import Any
import control as ct
import numpy as np

from .config import GanhosControlador, ParametrosSistema


def criar_planta(params: ParametrosSistema) -> ct.TransferFunction:
    """Constroi a planta linear G(s) = 1 / (s * (tau_p * s + 1))."""
    return ct.tf([1.0], [params.tau_p, 1.0, 0.0])


def criar_controlador_tf(
    ganhos: GanhosControlador, usar_filtro: bool = True
) -> ct.TransferFunction:
    """Retorna FT C(s) = Kp + Ki/s + Kd*s / (tau_f*s + 1)."""
    kp, ki, kd = ganhos.kp, ganhos.ki, ganhos.kd
    tf = ganhos.tau_f if usar_filtro else 0.0
    if ki == 0.0 and kd == 0.0:
        return ct.tf([kp], [1.0])
    if kd == 0.0:
        return ct.tf([kp, ki], [1.0, 0.0])
    if ki == 0.0:
        return ct.tf([kp * tf + kd, kp], [tf, 1.0]) if tf > 0 else ct.tf([kd, kp], [1.0])
    if tf > 0.0:
        return ct.tf([kp * tf + kd, kp + ki * tf, ki], [tf, 1.0, 0.0])
    return ct.tf([kd, kp, ki], [1.0, 0.0])


def verificar_estabilidade(polos: np.ndarray, tol: float = 1e-9) -> bool:
    """Retorna True se todos os polos estiverem no SPE aberto."""
    return bool(np.all(np.real(polos) < -tol))


def serializar_raizes(raizes: np.ndarray) -> list[dict[str, float]]:
    """Converte polos ou zeros complexos em lista serializavel."""
    return [{"re": round(float(r.real), 5), "im": round(float(r.imag), 5)}
            for r in raizes]


def calcular_margens(malha_aberta: ct.TransferFunction) -> dict[str, Any]:
    """Calcula margens de ganho (GM) e fase (PM) e frequencias criticas."""
    gm, pm, wcg, wcp = ct.margin(malha_aberta)
    fin_gm = gm is not None and np.isfinite(gm) and gm > 0
    gm_db = float(20.0 * np.log10(gm)) if fin_gm else None
    pm_v = float(pm) if (pm is not None and np.isfinite(pm)) else None
    wcg_v = float(wcg) if (wcg is not None and np.isfinite(wcg)) else None
    wcp_v = float(wcp) if (wcp is not None and np.isfinite(wcp)) else None
    return {
        "GM_abs": float(gm) if (gm is not None and np.isfinite(gm)) else "inf",
        "GM_dB": round(gm_db, 4) if gm_db is not None else "inf",
        "PM_deg": round(pm_v, 4) if pm_v is not None else None,
        "wcg_rads": round(wcg_v, 4) if wcg_v is not None else None,
        "wcp_rads": round(wcp_v, 4) if wcp_v is not None else None,
    }


def sintetizar_malha(
    ganhos: GanhosControlador, planta: ct.TransferFunction
) -> dict[str, Any]:
    """Sintetiza L(s)=C(s)G(s), T(s), polos, zeros, estabilidade e margens."""
    ctrl = criar_controlador_tf(ganhos, usar_filtro=(ganhos.kd > 0.0))
    malha_aberta = ct.minreal(ctrl * planta, verbose=False)
    malha_fechada = ct.feedback(malha_aberta, 1)
    polos_cl = ct.poles(malha_fechada)
    return {
        "C": ctrl, "L": malha_aberta, "T_cl": malha_fechada,
        "polos_cl": polos_cl, "zeros_cl": ct.zeros(malha_fechada),
        "estavel": verificar_estabilidade(polos_cl),
        "margens": calcular_margens(malha_aberta),
    }
