"""Geracao das figuras vetoriais (.pdf) e rasterizadas (.png) do estudo."""

from pathlib import Path
from typing import Sequence
import control as ct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CORES, GanhosControlador, ParametrosSistema
from .models import criar_controlador_tf
from .simulator import ResultadoSimulacao, simular_malha_aberta, simular_sistema

SeriePlot = tuple[str, str, str, ResultadoSimulacao]


def configurar_estilo() -> None:
    """Configura estilo visual padrao publicacao para todas as figuras."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.8, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linestyle": "--", "figure.dpi": 150,
    })


def salvar_figura(fig: plt.Figure, nome_base: str, dir_fig: Path) -> None:
    """Salva figura em PDF vetorial e PNG 200 dpi no diretorio alvo."""
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / f"{nome_base}.pdf", bbox_inches="tight")
    fig.savefig(dir_fig / f"{nome_base}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _formatar_eixo(
    ax: plt.Axes, *, titulo: str = "", xlabel: str = "", ylabel: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    leg_loc: str = "best", leg_ncol: int = 1, leg_fs: float | None = None,
) -> None:
    """Aplica rotulos, limites e legenda padronizados a um eixo."""
    if titulo:
        ax.set_title(titulo)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    kw = {"loc": leg_loc, "ncol": leg_ncol}
    if leg_fs is not None:
        kw["fontsize"] = leg_fs
    ax.legend(**kw)


def gerar_fig1_malha_aberta(
    params: ParametrosSistema, dir_fig: Path
) -> dict[str, dict[str, object]]:
    """Fig 1: Resposta em malha aberta ao degrau u(t) = U0 * 1(t)."""
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.2), sharex=True)
    dados_ma: dict[str, dict[str, object]] = {}
    for u0, cor in [(100.0, "#1f77b4"), (300.0, "#d62728")]:
        ma = simular_malha_aberta(u0, params)
        t = ma["t"]
        dados_ma[f"U0_{int(u0)}"] = {
            "t_esvaziamento_s": ma["t_esvaziamento_s"],
            "mu_final_evs": ma["mu_final_evs"],
        }
        axes[0].plot(t, ma["x_lin"], color=cor, ls="--", alpha=0.55)
        axes[0].plot(t, ma["x_fis"], color=cor,
                     label=f"$U_0 = {int(u0)}$ ev/s (físico, $q\\geq 0$)")
        axes[1].plot(t, ma["q_lin"], color=cor, ls="--", alpha=0.55)
        axes[1].plot(t, ma["q_fis"], color=cor, label=f"$U_0 = {int(u0)}$ ev/s")
        axes[2].plot(t, ma["mu"], color=cor,
                     label=f"$\\mu(t)$ para $U_0 = {int(u0)}$ ev/s")

    axes[0].axhline(params.x_ref, color="#333333", ls=":", label="$x_r = 300$ ev")
    axes[0].axhline(params.q_0, color="#666666", ls="-.", lw=1.1,
                    label="Limite $x=600$ ($q=0$)")
    _formatar_eixo(
        axes[0], ylabel="$x(t) = 600 - q(t)$ [ev]",
        titulo="Resposta em Malha Aberta: Redução da Fila, Fila Real e Capacidade",
        leg_loc="upper left", leg_ncol=2,
    )
    axes[1].axhline(params.q_ref, color="#333333", ls=":", label="$q_r = 300$ ev")
    axes[1].axhline(0.0, color="#666666", ls="-.", lw=1.1, label="Fila vazia ($q=0$)")
    _formatar_eixo(axes[1], ylabel="$q(t)$ [eventos]", ylim=(-120, 680),
                   leg_loc="upper right", leg_ncol=2)
    axes[2].axhline(params.mu_max, color="#888888", ls="-.", lw=1.1,
                    label="Limite sup. 1400 ev/s")
    _formatar_eixo(axes[2], xlabel="Tempo $t$ [s]", ylabel="$\\mu(t)$ [ev/s]",
                   ylim=(950, 1450), leg_loc="lower right", leg_ncol=2)
    fig.tight_layout()
    salvar_figura(fig, "fig1_malha_aberta", dir_fig)
    return dados_ma


def gerar_fig2_lgr_proporcional(
    k_vals: Sequence[float], dir_fig: Path
) -> None:
    """Fig 2: Lugar Geometrico das Raizes da planta com controlador P."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot([-0.5, 0.0], [0.0, 0.0], color="#1f77b4", lw=2.2,
            label="Ramos do LGR ($K > 0$)")
    ax.plot([-0.25, -0.25], [-1.25, 1.25], color="#1f77b4", lw=2.2)
    ax.axvline(-0.25, color="#999999", ls=":", lw=1.2,
               label="Assíntota $\\sigma = -0,25$ s$^{-1}$")
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.axvline(0.0, color="#444444", lw=0.8)
    ax.plot([0.0, -0.5], [0.0, 0.0], "rx", ms=9, mew=2.0,
            label="Polos de MA ($0; -0,5$)")

    estilo = [("o", "#2ca02c", (-18, 10)), ("s", "#ff7f0e", (-62, 10)),
              ("^", "#9467bd", (8, 6)), ("D", "#d62728", (8, 6))]
    for k, (mk, cor, ofs) in zip(k_vals, estilo):
        polos = np.roots([2.0, 1.0, k])
        ax.plot(polos.real, polos.imag, marker=mk, ls="none", color=cor,
                ms=7.5, label=f"$K = {k}$")
        p_ref = (polos[np.argmax(polos.imag)]
                 if np.any(np.abs(polos.imag) > 1e-6)
                 else polos[np.argmax(polos.real)])
        ax.annotate(f"$K={k}$", xy=(p_ref.real, p_ref.imag), xytext=ofs,
                    textcoords="offset points", fontsize=8.5,
                    color=cor, fontweight="bold")

    _formatar_eixo(
        ax, titulo="Lugar Geométrico das Raízes (LGR) — Planta com Controlador P",
        xlabel="Eixo Real $\\operatorname{Re}(s)$ [s$^{-1}$]",
        ylabel="Eixo Imaginário $\\operatorname{Im}(s)$ [rad/s]",
        xlim=(-0.75, 0.15), ylim=(-1.25, 1.25), leg_loc="lower left", leg_ncol=2,
    )
    fig.tight_layout()
    salvar_figura(fig, "fig2_lgr_proporcional", dir_fig)


def gerar_fig3_temporal_proporcional(
    k_vals: Sequence[float], params: ParametrosSistema, dir_fig: Path
) -> None:
    """Fig 3: Respostas temporais de x(t) e q(t) para controlador P."""
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)
    for k, cor in zip(k_vals, ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]):
        sim = simular_sistema(GanhosControlador(kp=k), params,
                              saturar=False, incluir_perturbacao=False)
        m = sim.t <= 35.0
        axes[0].plot(sim.t[m], sim.x[m], color=cor, label=f"$K = {k}$")
        axes[1].plot(sim.t[m], sim.q[m], color=cor, label=f"$K = {k}$")

    axes[0].axhline(params.x_ref, color="#333333", ls="--", lw=1.2,
                    label="$x_r = 300$ ev")
    axes[0].axhspan(0.98 * params.x_ref, 1.02 * params.x_ref, color="#aaaaaa",
                    alpha=0.2, label="Faixa $\\pm 2\\%$")
    axes[0].axhline(1.10 * params.x_ref, color="#777777", ls=":", lw=1.1,
                    label="Limite $M_p = 10\\%$ ($330$ ev)")
    _formatar_eixo(
        axes[0], ylabel="Redução da fila $x(t)$ [ev]",
        titulo="Controlador Proporcional: Resposta ao Degrau $x_r = 300$ ev",
        leg_loc="lower right", leg_ncol=2,
    )
    axes[1].axhline(params.q_ref, color="#333333", ls="--", lw=1.2,
                    label="$q_r = 300$ ev")
    _formatar_eixo(axes[1], xlabel="Tempo $t$ [s]", ylabel="Fila real $q(t)$ [ev]",
                   leg_loc="upper right", leg_ncol=2)
    fig.tight_layout()
    salvar_figura(fig, "fig3_temporal_proporcional", dir_fig)


def gerar_fig4_lgr_compensadores(
    configs_lgr: Sequence[tuple[str, ct.TransferFunction, str]],
    planta: ct.TransferFunction, dir_fig: Path,
) -> None:
    """Fig 4: Painel 2x2 com LGR dos sistemas compensados P, PI, PD e PID."""
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.0))
    kv = np.concatenate([np.linspace(0.0, 0.2, 400), np.linspace(0.2, 2.5, 1000),
                         np.linspace(2.5, 15.0, 600)])
    for ax, (titulo, ctrl, cor) in zip(axes.flat, configs_lgr):
        malha = ct.minreal(ctrl * planta, verbose=False)
        num_l = np.asarray(malha.num[0][0], dtype=float)
        den_l = np.asarray(malha.den[0][0], dtype=float)
        deg = max(len(num_l), len(den_l)) - 1
        num_p = np.pad(num_l, (deg + 1 - len(num_l), 0))
        den_p = np.pad(den_l, (deg + 1 - len(den_l), 0))
        raizes = np.array([np.roots(den_p + k * num_p) for k in kv])
        for col in range(raizes.shape[1]):
            ax.plot(raizes[:, col].real, raizes[:, col].imag, ".",
                    color=cor, ms=1.5, alpha=0.8)
        p_ma, z_ma = ct.poles(malha), ct.zeros(malha)
        p_mf = np.roots(den_p + num_p)
        ax.axhline(0.0, color="#444444", lw=0.7)
        ax.axvline(0.0, color="#444444", lw=0.7)
        ax.plot(p_ma.real, p_ma.imag, "kx", ms=8, mew=1.8, label="Polos MA")
        if len(z_ma) > 0:
            ax.plot(z_ma.real, z_ma.imag, "ko", mfc="none", ms=8, mew=1.6,
                    label="Zeros MA")
        ax.plot(p_mf.real, p_mf.imag, "r*", ms=10, label="Polos MF ($K=1$)")
        _formatar_eixo(
            ax, titulo=titulo, xlabel="$\\operatorname{Re}(s)$ [s$^{-1}$]",
            ylabel="$\\operatorname{Im}(s)$ [rad/s]", xlim=(-2.2, 0.4),
            ylim=(-1.6, 1.6), leg_loc="upper left", leg_fs=7.8,
        )
    fig.suptitle("Lugar Geométrico das Raízes dos Sistemas Compensados",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    salvar_figura(fig, "fig4_lgr_compensadores", dir_fig)


def gerar_fig5_bode_compensadores(
    configs_bode: Sequence[tuple[str, ct.TransferFunction, str]],
    planta: ct.TransferFunction, dir_fig: Path,
) -> None:
    """Fig 5: Diagramas de Bode de L(jw) para P, PI, PD e PID."""
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    w = np.logspace(-2, 2, 1000)
    for nome, ctrl, cor in configs_bode:
        malha = ct.minreal(ctrl * planta, verbose=False)
        mag, phase, omega = ct.frequency_response(malha, w)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
        ph_deg = np.unwrap(phase) * 180.0 / np.pi
        while ph_deg[0] > 0.0:
            ph_deg -= 360.0
        while ph_deg[0] < -270.0:
            ph_deg += 360.0
        gm, pm, _, wcp = ct.margin(malha)
        lbl = f"{nome} (PM={pm:.1f}$^\\circ$" if np.isfinite(pm) else f"{nome} (PM=N/D"
        lbl += (", GM=$\\infty$)" if not np.isfinite(gm)
                else f", GM={20*np.log10(gm):.1f} dB)")
        ax_mag.semilogx(omega, mag_db, color=cor, label=lbl)
        ax_ph.semilogx(omega, ph_deg, color=cor, label=nome)
        if wcp is not None and np.isfinite(wcp) and np.isfinite(pm):
            ax_mag.plot(wcp, 0.0, "o", color=cor, ms=5.5)
            ax_ph.plot(wcp, -180.0 + pm, "o", color=cor, ms=5.5)

    ax_mag.axhline(0.0, color="#333333", ls="--", lw=1.1, label="Cruzamento 0 dB")
    _formatar_eixo(
        ax_mag, ylabel="Magnitude [dB]", ylim=(-60, 60),
        titulo="Diagramas de Bode de Malha Aberta Compensada "
               "$L(j\\omega) = C(j\\omega)G(j\\omega)$",
        leg_loc="upper right", leg_fs=8.2,
    )
    ax_ph.axhline(-180.0, color="#333333", ls="--", lw=1.1,
                  label="Limite $-180^\\circ$")
    ax_ph.axhline(-135.0, color="#777777", ls=":", lw=1.0,
                  label="PM $= 45^\\circ$ ($-135^\\circ$)")
    _formatar_eixo(
        ax_ph, xlabel="Frequência $\\omega$ [rad/s]", ylabel="Fase [graus]",
        ylim=(-210, -45), leg_loc="lower left", leg_ncol=2, leg_fs=8.2,
    )
    fig.tight_layout()
    salvar_figura(fig, "fig5_bode_compensadores", dir_fig)


def gerar_fig6_resposta_completa_60s(
    sims_lin: Sequence[SeriePlot], sims_sat: Sequence[SeriePlot],
    params: ParametrosSistema, dir_fig: Path,
) -> None:
    """Fig 6: Comparacao temporal (0..60 s) linear vs saturada com perturbacao."""
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex=True)
    paineis = [("Regime Linear (sem saturação)", sims_lin, 0),
               ("Regime Não Linear (saturação [600, 1400] ev/s)", sims_sat, 1)]
    for titulo, sims, col in paineis:
        ax_x, ax_q = axes[0, col], axes[1, col]
        for label, cor, ls, sim in sims:
            ax_x.plot(sim.t, sim.x, color=cor, ls=ls, label=label)
            ax_q.plot(sim.t, sim.q, color=cor, ls=ls, label=label)
        ax_x.axhline(params.x_ref, color="#222222", ls=":", lw=1.2,
                     label="$x_r = 300$")
        ax_x.axvline(params.t_dist, color="#666666", ls="-.", lw=1.0)
        ax_x.axhspan(294.0, 306.0, color="#bbbbbb", alpha=0.2)
        _formatar_eixo(
            ax_x, titulo=titulo, ylim=(-20, 380), leg_loc="lower left", leg_fs=7.6,
            ylabel="Redução da fila $x(t)$ [ev]" if col == 0 else "",
        )
        ax_q.axhline(params.q_ref, color="#222222", ls=":", lw=1.2,
                     label="$q_r = 300$")
        ax_q.axvline(params.t_dist, color="#666666", ls="-.", lw=1.0,
                     label="Perturbação (+100 ev/s)")
        _formatar_eixo(
            ax_q, xlabel="Tempo $t$ [s]", ylim=(220, 620), leg_loc="upper left",
            leg_fs=7.6, ylabel="Fila real $q(t)$ [ev]" if col == 0 else "",
        )
    fig.suptitle(
        "Comparação Temporal (0 a 60 s): Rastreamento e Rejeição a Perturbação",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    salvar_figura(fig, "fig6_resposta_completa_60s", dir_fig)


def gerar_fig7_capacidade_saturacao(
    sims_sat: Sequence[SeriePlot], params: ParametrosSistema, dir_fig: Path
) -> None:
    """Fig 7: Capacidade solicitada mu_c(t) e efetiva mu(t) vs limites fisicos."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6))
    for label, cor, ls, sim in sims_sat:
        mz = sim.t <= 6.0
        axes[0, 0].plot(sim.t, sim.muc, color=cor, ls=ls, label=label)
        axes[0, 1].plot(sim.t[mz], sim.muc[mz], color=cor, ls=ls, label=label)
        axes[1, 0].plot(sim.t, sim.mu, color=cor, ls=ls, label=label)
        axes[1, 1].plot(sim.t[mz], sim.mu[mz], color=cor, ls=ls, label=label)
    for j in range(2):
        axes[0, j].axhline(params.mu_max, color="#333333", ls="--", lw=1.1,
                           label="Lim. 1400")
        axes[0, j].axhline(params.mu_min, color="#333333", ls="--", lw=1.1,
                           label="Lim. 600")
        axes[0, j].set_ylim(550, 1850)
        axes[1, j].axhline(params.mu_max, color="#333333", ls="--", lw=1.1,
                           label="Lim. 1400")
        axes[1, j].set_ylim(950, 1430)
        axes[1, j].set_xlabel("Tempo $t$ [s]")
    axes[0, 0].axvline(params.t_dist, color="#666666", ls="-.", lw=1.0)
    axes[1, 0].axvline(params.t_dist, color="#666666", ls="-.", lw=1.0)
    axes[0, 0].set_ylabel("$\\mu_c(t)$ solicitada [ev/s]")
    axes[1, 0].set_ylabel("$\\mu(t)$ efetiva [ev/s]")
    axes[0, 0].set_title("Horizonte Completo (0 a 60 s)")
    axes[0, 1].set_title("Zoom na Saturação Inicial (0 a 6 s)")
    axes[0, 0].legend(loc="upper right", fontsize=7.4, ncol=2)
    axes[1, 0].legend(loc="upper right", fontsize=7.4, ncol=2)
    fig.suptitle(
        "Capacidade Solicitada vs. Efetiva e Limites Físicos [600, 1400] ev/s",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    salvar_figura(fig, "fig7_capacidade_saturacao", dir_fig)


def gerar_fig8_pid_windup_derivativa(
    pid_vars: Sequence[SeriePlot], pi_lin: ResultadoSimulacao,
    pi_sat: ResultadoSimulacao, params: ParametrosSistema, dir_fig: Path,
) -> None:
    """Fig 8: Efeitos de windup, derivative kick e caso PI instavel."""
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    for label, cor, ls, sim in pid_vars:
        m30, m15 = sim.t <= 30.0, sim.t <= 1.5
        axes[0, 0].plot(sim.t[m30], sim.x[m30], color=cor, ls=ls, label=label)
        axes[0, 1].plot(sim.t[m15], sim.muc[m15], color=cor, ls=ls, label=label)

    axes[0, 0].axhline(params.x_ref, color="#222222", ls=":", lw=1.2)
    axes[0, 0].axhline(330.0, color="#888888", ls="--", lw=1.0,
                       label="Limite $M_p=10\\%$")
    axes[0, 0].axvline(params.t_dist, color="#666666", ls="-.", lw=1.0)
    _formatar_eixo(
        axes[0, 0], titulo="PID: Windup, Anti-Windup e Derivada na Medição",
        xlabel="Tempo $t$ [s]", ylabel="$x(t)$ [ev]",
        leg_loc="lower right", leg_fs=7.3,
    )
    axes[0, 1].axhline(params.mu_max, color="#333333", ls="--", lw=1.2,
                       label="Limite sup. 1400")
    axes[0, 1].set_yscale("log")
    _formatar_eixo(
        axes[0, 1], titulo="Pico Derivativo (Derivative Kick) em $\\mu_c(t)$",
        xlabel="Tempo $t$ [s]", ylabel="$\\mu_c(t)$ [ev/s] (escala log)",
        leg_loc="upper right", leg_fs=7.3,
    )
    axes[1, 0].plot(pi_lin.t, pi_lin.x, color="#d62728", ls="--",
                    label="PI instável linear ($K_p=0,50, K_i=0,30$)")
    axes[1, 0].plot(pi_sat.t, pi_sat.x, color="#1f77b4", ls="-",
                    label="PI instável com saturação e $q \\geq 0$")
    axes[1, 0].axhline(params.x_ref, color="#222222", ls=":", lw=1.2,
                       label="$x_r = 300$")
    _formatar_eixo(
        axes[1, 0], titulo="PI Instável ($K_i = 0,30 > K_p/2 = 0,25$): Divergência",
        xlabel="Tempo $t$ [s]", ylabel="$x(t)$ [ev]",
        ylim=(-900, 1100), leg_loc="lower left", leg_fs=7.6,
    )
    axes[1, 1].plot(pi_sat.t, pi_sat.mu, color="#1f77b4",
                    label="$\\mu(t)$ efetiva")
    axes[1, 1].axhline(params.mu_max, color="#333333", ls="--", lw=1.1,
                       label="Limites [600, 1400]")
    axes[1, 1].axhline(params.mu_min, color="#333333", ls="--", lw=1.1)
    _formatar_eixo(
        axes[1, 1], titulo="PI Instável Saturado: Ciclo Limite em $\\mu(t)$",
        xlabel="Tempo $t$ [s]", ylabel="$\\mu(t)$ [ev/s]",
        ylim=(550, 1450), leg_loc="upper right", leg_fs=7.6,
    )
    fig.suptitle(
        "Evidências Computacionais das Limitações: Windup, Kick e Instabilidade",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    salvar_figura(fig, "fig8_pid_windup_derivativa", dir_fig)


def gerar_todas_figuras(
    k_vals: Sequence[float], ganhos_comp: dict[str, GanhosControlador],
    sims_lin: dict[str, ResultadoSimulacao],
    sims_sat: dict[str, ResultadoSimulacao], planta: ct.TransferFunction,
    params: ParametrosSistema, dir_fig: Path,
) -> dict[str, dict[str, object]]:
    """Gera o conjunto completo das 8 figuras em PDF e PNG."""
    configurar_estilo()
    dados_ma = gerar_fig1_malha_aberta(params, dir_fig)
    gerar_fig2_lgr_proporcional(k_vals, dir_fig)
    gerar_fig3_temporal_proporcional(k_vals, params, dir_fig)

    c_p = criar_controlador_tf(ganhos_comp["P"])
    c_pi = criar_controlador_tf(ganhos_comp["PI"])
    c_pd = criar_controlador_tf(ganhos_comp["PD"])
    c_pid = criar_controlador_tf(ganhos_comp["PID_nom"])
    c_pid_adj = criar_controlador_tf(ganhos_comp["PID_adj"])

    gerar_fig4_lgr_compensadores([
        (f"P ($K={ganhos_comp['P'].kp}$)", c_p, CORES["P"]),
        (f"PI ($K_p={ganhos_comp['PI'].kp}, K_i={ganhos_comp['PI'].ki}$)",
         c_pi, CORES["PI"]),
        (f"PD ($K_p={ganhos_comp['PD'].kp}, K_d={ganhos_comp['PD'].kd}, "
         f"\\tau_f=0,05$)", c_pd, CORES["PD"]),
        (f"PID ($K_p={ganhos_comp['PID_nom'].kp}, "
         f"K_i={ganhos_comp['PID_nom'].ki}, K_d={ganhos_comp['PID_nom'].kd}$)",
         c_pid, CORES["PID"]),
    ], planta, dir_fig)

    gerar_fig5_bode_compensadores([
        ("P ($K=0,369$)", c_p, CORES["P"]),
        ("PI ($0,35; 0,03$)", c_pi, CORES["PI"]),
        ("PD ($1,28; 1,56$)", c_pd, CORES["PD"]),
        ("PID nom. ($\\alpha=0,15$)", c_pid, CORES["PID"]),
        ("PID ajust. ($\\alpha=0,40$)", c_pid_adj, CORES["PID_adj"]),
    ], planta, dir_fig)

    gerar_fig6_resposta_completa_60s(
        sims_lin=[
            ("P ($K=0,369$)", CORES["P"], "-", sims_lin["P_ref"]),
            ("PI ($0,35; 0,03$)", CORES["PI"], "-", sims_lin["PI_estavel"]),
            ("PD ($1,28; 1,56$)", CORES["PD"], "-", sims_lin["PD_filt"]),
            ("PID nom. ($\\alpha=0,15$)", CORES["PID"], "-",
             sims_lin["PID_nom_filt"]),
        ],
        sims_sat=[
            ("P ($K=0,369$)", CORES["P"], "-", sims_sat["P_ref_sat"]),
            ("PI ($0,35; 0,03$)", CORES["PI"], "-",
             sims_sat["PI_estavel_sat_noAW"]),
            ("PD (D-med)", CORES["PD"], "-", sims_sat["PD_filt_sat_Dmeas"]),
            ("PID nom. (AW+D-med)", CORES["PID"], "--",
             sims_sat["PID_nom_sat_backcalc_Dmeas"]),
            ("PID ajust. ($\\alpha=0,40$, AW)", CORES["PID_adj"], "-",
             sims_sat["PID_adj_sat_backcalc_Dmeas"]),
        ],
        params=params, dir_fig=dir_fig,
    )
    gerar_fig7_capacidade_saturacao([
        ("P ($K=0,369$)", CORES["P"], "-", sims_sat["P_ref_sat"]),
        ("PI ($0,35; 0,03$)", CORES["PI"], "-", sims_sat["PI_estavel_sat_noAW"]),
        ("PD (D-err)", CORES["PD"], ":", sims_sat["PD_filt_sat_Derr"]),
        ("PD (D-med)", CORES["PD"], "-", sims_sat["PD_filt_sat_Dmeas"]),
        ("PID nom. (sem AW, D-err)", "#8c564b", ":",
         sims_sat["PID_nom_sat_noAW_Derr"]),
        ("PID ajust. (AW, D-med)", CORES["PID_adj"], "-",
         sims_sat["PID_adj_sat_backcalc_Dmeas"]),
    ], params, dir_fig)
    gerar_fig8_pid_windup_derivativa(
        pid_vars=[
            ("PID nom. linear (D-erro)", "#444444", "--",
             sims_lin["PID_nom_filt"]),
            ("PID nom. sat. sem AW (D-erro)", "#d62728", "-",
             sims_sat["PID_nom_sat_noAW_Derr"]),
            ("PID nom. sat. clamp + D-med", "#ff7f0e", "-.",
             sims_sat["PID_nom_sat_clamp_Dmeas"]),
            ("PID nom. sat. back-calc + D-med", "#1f77b4", "-",
             sims_sat["PID_nom_sat_backcalc_Dmeas"]),
            ("PID ajust. ($\\alpha=0,40$) back-calc + D-med",
             CORES["PID_adj"], "-", sims_sat["PID_adj_sat_backcalc_Dmeas"]),
        ],
        pi_lin=sims_lin["PI_instavel"], pi_sat=sims_sat["PI_instavel_sat"],
        params=params, dir_fig=dir_fig,
    )
    return dados_ma
