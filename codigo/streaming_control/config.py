"""Configuracoes, dataclasses imutaveis e caminhos do projeto."""

from dataclasses import dataclass
from pathlib import Path

RAIZ_PROJETO: Path = Path(__file__).resolve().parents[2]
DIR_RESULTADOS: Path = RAIZ_PROJETO / "resultados"
DIR_FIGURAS: Path = RAIZ_PROJETO / "relatorio" / "figuras"

CORES: dict[str, str] = {
    "P": "#1f77b4", "PI": "#ff7f0e", "PD": "#2ca02c",
    "PID": "#d62728", "PID_adj": "#9467bd",
}


@dataclass(frozen=True)
class RequisitosDesempenho:
    """Especificacoes de desempenho em malha fechada."""

    tr_max_s: float = 5.0
    ts_max_s: float = 12.0
    mp_max_pct: float = 10.0
    ess_max_ev: float = 10.0
    pm_min_deg: float = 45.0
    mu_min_evs: float = 600.0
    mu_max_evs: float = 1400.0

    def to_dict(self) -> dict[str, float]:
        """Serializa os requisitos para JSON."""
        return {
            "tr_max_s": self.tr_max_s, "ts_max_s": self.ts_max_s,
            "Mp_max_pct": self.mp_max_pct, "ess_max_ev": self.ess_max_ev,
            "PM_min_deg": self.pm_min_deg, "mu_min_evs": self.mu_min_evs,
            "mu_max_evs": self.mu_max_evs,
        }


@dataclass(frozen=True)
class ParametrosSistema:
    """Parametros nominais do sistema de streaming e cenario de ensaio."""

    lambda_0: float = 1000.0
    mu_0: float = 1000.0
    q_0: float = 600.0
    q_ref: float = 300.0
    tau_p: float = 2.0
    u_min: float = -400.0
    u_max: float = 400.0
    t_dist: float = 20.0
    dlambda_step: float = 100.0
    t_end: float = 60.0
    tau_f: float = 0.05

    @property
    def x_ref(self) -> float:
        """Reducao de fila desejada x_r = q(0) - q_ref."""
        return self.q_0 - self.q_ref

    @property
    def mu_min(self) -> float:
        """Capacidade fisica minima (ev/s)."""
        return self.mu_0 + self.u_min

    @property
    def mu_max(self) -> float:
        """Capacidade fisica maxima (ev/s)."""
        return self.mu_0 + self.u_max

    def to_dict(self, reqs: RequisitosDesempenho) -> dict[str, object]:
        """Serializa os parametros e requisitos do estudo."""
        return {
            "lambda_0_evs": self.lambda_0, "mu_0_evs": self.mu_0,
            "q_0_ev": self.q_0, "q_ref_ev": self.q_ref,
            "x_ref_ev": self.x_ref, "tau_p_s": self.tau_p,
            "mu_min_evs": self.mu_min, "mu_max_evs": self.mu_max,
            "t_dist_s": self.t_dist, "dlambda_step_evs": self.dlambda_step,
            "t_end_s": self.t_end, "tau_f_s": self.tau_f,
            "requisitos": reqs.to_dict(),
        }


@dataclass(frozen=True)
class GanhosControlador:
    """Ganhos P, I, D e filtro derivativo realizavel."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    tau_f: float = 0.05

    @classmethod
    def alocar_pd(
        cls, wn: float, zeta: float, tau_f: float = 0.05
    ) -> "GanhosControlador":
        """Sintonia PD por alocacao de 2a ordem em G(s)=1/(s(2s+1))."""
        return cls(
            kp=round(2.0 * wn**2, 4),
            kd=round(4.0 * zeta * wn - 1.0, 4),
            tau_f=tau_f,
        )

    @classmethod
    def alocar_pid(
        cls, wn: float, zeta: float, alpha: float, tau_f: float = 0.05
    ) -> "GanhosControlador":
        """Sintonia PID por alocacao (s+alpha)(s^2+2*zeta*wn*s+wn^2)."""
        return cls(
            kp=round(2.0 * (wn**2 + 2.0 * alpha * zeta * wn), 4),
            ki=round(2.0 * alpha * wn**2, 4),
            kd=round(2.0 * (alpha + 2.0 * zeta * wn) - 1.0, 4),
            tau_f=tau_f,
        )

    def to_dict(self) -> dict[str, float]:
        """Retorna ganhos arredondados para registro."""
        return {
            "Kp": round(float(self.kp), 4),
            "Ki": round(float(self.ki), 4),
            "Kd": round(float(self.kd), 4),
            "tau_f": round(float(self.tau_f), 4) if self.kd > 0.0 else 0.0,
        }


PARAMETROS_PADRAO = ParametrosSistema()
REQUISITOS_PADRAO = RequisitosDesempenho()
