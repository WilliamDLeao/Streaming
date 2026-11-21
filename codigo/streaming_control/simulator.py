"""Integrador temporal RK4 para regimes linear e nao linear."""

from dataclasses import dataclass
import numpy as np

from .config import GanhosControlador, ParametrosSistema


@dataclass(frozen=True)
class ConfigSimulacao:
    """Define um cenario de simulacao para um controlador."""

    nome: str
    ganhos: GanhosControlador
    saturar: bool = True
    anti_windup: str = "none"
    deriv_na_medicao: bool = False
    tt_aw: float = 0.5
    incluir_tf_ideal: bool = False


@dataclass(frozen=True)
class ResultadoSimulacao:
    """Series temporais produzidas por uma simulacao."""

    t: np.ndarray
    x: np.ndarray
    q: np.ndarray
    dmu: np.ndarray
    u_cmd: np.ndarray
    u_eff: np.ndarray
    muc: np.ndarray
    mu: np.ndarray


def simular_malha_aberta(
    u0: float, params: ParametrosSistema, t_max: float = 12.0, n_pts: int = 1201
) -> dict[str, object]:
    """Calcula a resposta analitica em malha aberta ao degrau u(t) = U0*1(t)."""
    t = np.linspace(0.0, t_max, n_pts)
    x_lin = u0 * (t - 2.0 + 2.0 * np.exp(-0.5 * t))
    q_lin = params.q_0 - x_lin
    mu_t = params.mu_0 + u0 * (1.0 - np.exp(-0.5 * t))
    idx_zero = int(np.argmax(q_lin <= 0.0))
    t_vazio = float(t[idx_zero]) if q_lin[idx_zero] <= 0.0 else None
    return {
        "t": t, "x_lin": x_lin, "q_lin": q_lin,
        "x_fis": np.minimum(x_lin, params.q_0), "q_fis": np.maximum(q_lin, 0.0),
        "mu": mu_t,
        "t_esvaziamento_s": round(t_vazio, 4) if t_vazio is not None else None,
        "mu_final_evs": round(float(mu_t[-1]), 2),
    }


def simular_sistema(
    ganhos: GanhosControlador,
    params: ParametrosSistema,
    *,
    saturar: bool = True,
    anti_windup: str = "none",
    deriv_na_medicao: bool = False,
    tt_aw: float = 0.5,
    incluir_perturbacao: bool = True,
    dt: float = 0.002,
) -> ResultadoSimulacao:
    """
    Integra o sistema em malha fechada via RK4 (estados: [x, dmu, xi, z_d]).

    Sob saturacao fisica, aplica u_sat em [u_min, u_max] e limita q >= 0
    (x <= q_0). Suporta anti-windup ('clamp' ou 'backcalc') e derivada na
    medicao (sig = -x) para suprimir o pico derivativo no degrau inicial.
    """
    n_pts = int(round(params.t_end / dt)) + 1
    t = np.linspace(0.0, params.t_end, n_pts)
    kp, ki, kd, tau_f = ganhos.kp, ganhos.ki, ganhos.kd, ganhos.tau_f

    x_arr, q_arr, dmu_arr = np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts)
    muc_arr, mu_arr = np.zeros(n_pts), np.zeros(n_pts)
    u_cmd_arr, u_eff_arr = np.zeros(n_pts), np.zeros(n_pts)
    s = np.zeros(4)

    def avaliar_controle(s_vec: np.ndarray) -> tuple[float, float, float, float]:
        x_val, _, xi_val, zd_val = s_vec
        e_val = params.x_ref - x_val
        # Derivada na medicao evita o salto brusco do erro no degrau em t=0
        sig = -x_val if deriv_na_medicao else e_val
        deriv = (sig - zd_val) / tau_f if (kd > 0.0 and tau_f > 0.0) else 0.0
        u_unsat = kp * e_val + ki * xi_val + kd * deriv
        u_sat = (float(np.clip(u_unsat, params.u_min, params.u_max))
                 if saturar else u_unsat)
        return u_unsat, u_sat, e_val, sig

    def derivadas(tk: float, s_vec: np.ndarray) -> np.ndarray:
        x_val, dmu_val, _, zd_val = s_vec
        dlam = (params.dlambda_step
                if (incluir_perturbacao and tk >= params.t_dist) else 0.0)
        u_unsat, u_sat, e_val, sig = avaliar_controle(s_vec)

        # Restricao fisica: fila nao negativa (q >= 0 <=> x <= q_0)
        dx = 0.0 if (saturar and x_val >= params.q_0 and dmu_val - dlam > 0.0) else (
            dmu_val - dlam
        )
        ddmu = (u_sat - dmu_val) / params.tau_p

        if ki <= 0.0:
            dxi = 0.0
        elif anti_windup == "clamp" and saturar:
            sat_dir = ((u_unsat > params.u_max and e_val > 0.0)
                       or (u_unsat < params.u_min and e_val < 0.0))
            dxi = 0.0 if sat_dir else e_val
        elif anti_windup == "backcalc" and saturar:
            # Descarrega o integrador proporcionalmente a diferenca (u_sat - u)
            dxi = e_val + (1.0 / (ki * tt_aw)) * (u_sat - u_unsat)
        else:
            dxi = e_val

        dzd = (sig - zd_val) / tau_f if (kd > 0.0 and tau_f > 0.0) else 0.0
        return np.array([dx, ddmu, dxi, dzd])

    for k in range(n_pts):
        tk = t[k]
        u_unsat, u_sat, _, _ = avaliar_controle(s)
        x_arr[k], q_arr[k], dmu_arr[k] = s[0], params.q_0 - s[0], s[1]
        u_cmd_arr[k], u_eff_arr[k] = u_unsat, u_sat
        muc_arr[k], mu_arr[k] = params.mu_0 + u_unsat, params.mu_0 + s[1]
        if k < n_pts - 1:
            k1 = derivadas(tk, s)
            k2 = derivadas(tk + 0.5 * dt, s + 0.5 * dt * k1)
            k3 = derivadas(tk + 0.5 * dt, s + 0.5 * dt * k2)
            k4 = derivadas(tk + dt, s + dt * k3)
            s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if saturar and s[0] > params.q_0:
                s[0] = params.q_0

    return ResultadoSimulacao(t=t, x=x_arr, q=q_arr, dmu=dmu_arr,
                              u_cmd=u_cmd_arr, u_eff=u_eff_arr,
                              muc=muc_arr, mu=mu_arr)
