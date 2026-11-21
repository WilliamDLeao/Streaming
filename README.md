# Modelagem Dinâmica e Controle de Desempenho de um Sistema de Processamento de Dados em *Streaming*

Projeto completo de modelagem física e matemática, análise dinâmica em malha aberta e malha fechada, projeto de compensadores clássicos (P, PI, PD e PID) e simulação temporal não linear aplicados à regulação de fila em uma plataforma contínua de processamento de eventos IoT em *streaming*.

---

## Visão Geral do Problema

A plataforma recebe eventos a uma taxa nominal $\lambda_0 = 1000\text{ eventos/s}$, possui fila inicial $q(0) = 600\text{ eventos}$ e ajusta a capacidade de processamento $\mu(t)$ em torno de $\mu_0 = 1000\text{ eventos/s}$ com dinâmica de primeira ordem ($\tau = 2{,}0\text{ s}$) limitada ao intervalo físico $[600,\,1400]\text{ eventos/s}$. Em variáveis incrementais $x(t) = 600 - q(t)$ e $u(t) = \mu_c(t) - 1000$, a planta é descrita por:

$$G(s) = \frac{X(s)}{U(s)} = \frac{1}{s(2s + 1)}$$

O controlador deve reduzir a fila para $q_r = 300\text{ eventos}$ ($x_r = 300\text{ eventos}$), rejeitar uma perturbação permanente em degrau $\Delta\lambda = +100\text{ eventos/s}$ aplicada em $t = 20\text{ s}$ e satisfazer simultaneamente seis especificações operacionais:

- Tempo de subida: $t_r(10\%\text{--}90\%) \le 5{,}0\text{ s}$
- Tempo de acomodação: $t_s(2\%) \le 12{,}0\text{ s}$
- Sobressinal máximo: $M_p \le 10{,}0\%$
- Erro em regime permanente (referência e pós-perturbação): $e_{ss} \le 10{,}0\text{ eventos}$
- Margem de fase mínima: $\mathrm{PM} \ge 45{,}0^\circ$
- Capacidade efetiva dentro dos limites físicos: $\mu(t) \in [600,\,1400]\text{ eventos/s}$

---

## Estrutura de Diretórios

```text
Controle/
├── Diretrizes_trabalho.pdf                 # Enunciado de referência do projeto
├── README.md                               # Documentação geral, arquitetura e instruções
├── codigo/                                 # Pacote Python modularizado de simulação
│   ├── main.py                             # Orquestrador CLI (métricas, JSON e figuras)
│   ├── requirements.txt                    # Dependências Python do projeto
│   └── streaming_control/
│       ├── __init__.py                     # Exportação pública do pacote
│       ├── config.py                       # Dataclasses de parâmetros, ganhos e caminhos
│       ├── models.py                       # Planta G(s), FTs C(s), polos, zeros e margens
│       ├── simulator.py                    # Integrador RK4 linear e não linear
│       ├── evaluation.py                   # Indicadores transitórios, erro e requisitos
│       └── plotting.py                     # Geração das 8 figuras vetoriais (PDF e PNG)
├── resultados/
│   └── resultados_metricas.json            # Saída consolidada de polos, margens e métricas
└── relatorio/                              # Relatório técnico-científico ABNT em LaTeX
    ├── main.tex                            # Documento mestre (máximo 30 páginas)
    ├── main.pdf                            # Relatório final compilado (30 páginas)
    ├── figuras/                            # 8 figuras em PDF vetorial e PNG (200 dpi)
    │   ├── fig1_malha_aberta.{pdf,png}
    │   ├── fig2_lgr_proporcional.{pdf,png}
    │   ├── fig3_temporal_proporcional.{pdf,png}
    │   ├── fig4_lgr_compensadores.{pdf,png}
    │   ├── fig5_bode_compensadores.{pdf,png}
    │   ├── fig6_resposta_completa_60s.{pdf,png}
    │   ├── fig7_capacidade_saturacao.{pdf,png}
    │   └── fig8_pid_windup_derivativa.{pdf,png}
    ├── preambulo/
    │   ├── pacotes_e_estilo.tex            # Pacotes, tipografia, ABNT, captions e listings
    │   └── pre_textuais.tex                # Capa, folha de rosto, resumo, listas e sumário
    ├── secoes/
    │   ├── 01_introducao.tex               # Contextualização, problema e organização
    │   ├── 02_modelagem_fisica.tex         # Esquema físico, diagrama de blocos TikZ e EDOs
    │   ├── 03_malha_aberta.tex             # EDO de 2ª ordem, FT G(s) e resposta em rampa
    │   ├── 04_controle_proporcional.tex    # Routh-Hurwitz, análise modal e LGR do controle P
    │   ├── 05_projeto_compensadores.tex    # Projeto P, PI, PD e PID, Bode e limitações
    │   ├── 06_simulacao_comparativa.tex    # Simulações de 60 s, saturação e Tabela 4
    │   └── 07_conclusoes.tex               # Trade-offs de engenharia e considerações finais
    └── pos_textuais/
        ├── referencias.tex                 # Referências bibliográficas (ABNT NBR 6023)
        └── apendice_codigo.tex             # Inclusão organizada dos módulos de codigo/
```

---

## Arquitetura Modular

### Código Python (`codigo/`)

O código de simulação segue separação explícita de responsabilidades, tipagem estática e imutabilidade de configurações via `@dataclass(frozen=True)`:

1. **`streaming_control/config.py`**: centraliza caminhos absolutos independentes do diretório de chamada (`pathlib.Path(__file__).resolve()`), paleta de cores (`CORES`) e as dataclasses `ParametrosSistema`, `RequisitosDesempenho` e `GanhosControlador` (incluindo os métodos de alocação analítica `alocar_pd` e `alocar_pid`).
2. **`streaming_control/models.py`**: constrói a planta `criar_planta`, sintetiza as funções de transferência de controladores ideais e filtrados (`criar_controlador_tf`), verifica estabilidade assintótica (`verificar_estabilidade`), calcula margens de ganho e fase (`calcular_margens`) e gera o modelo de malha fechada (`sintetizar_malha`).
3. **`streaming_control/simulator.py`**: implementa a resposta analítica em malha aberta (`simular_malha_aberta`) e o integrador temporal Runge-Kutta de 4ª ordem (`simular_sistema`, passo `dt = 0,002 s`), contemplando saturação física em $[600,\,1400]\text{ eventos/s}$, fronteira não negativa da fila ($q(t) \ge 0$), *anti-windup* (`none`, `clamp` e `backcalc` com $T_t = 0{,}5\text{ s}$) e ação derivativa aplicada ao erro ou à medição.
4. **`streaming_control/evaluation.py`**: extrai indicadores de degrau (`calcular_metricas_degrau`: $t_r$, $t_s$, $M_p$, $e_{ss,\text{ref}}$), de rejeição à perturbação (`calcular_metricas_perturbacao`), de esforço do atuador (`calcular_metricas_capacidade`) e verifica atendimento individual e global às metas (`avaliar_requisitos`, `analisar_configuracao`).
5. **`streaming_control/plotting.py`**: gera as 8 figuras comparativas (`gerar_fig1_malha_aberta` a `gerar_fig8_pid_windup_derivativa`) em PDF vetorial e PNG (`200 dpi`) dentro de `relatorio/figuras/`.
6. **`main.py`**: define a varredura proporcional e os cenários lineares e não lineares (`construir_cenarios`), executa todas as análises, aciona a geração das figuras e grava `resultados/resultados_metricas.json`.

### Relatório LaTeX (`relatorio/`)

O texto técnico-científico segue as normas da ABNT (NBR 14724, NBR 6023, NBR 6024, NBR 6027 e NBR 6028), com espaçamento 1,5, margens `3,0 / 2,0 cm`, diagramas vetoriais em TikZ e código-fonte Python integrado via `listings` no Apêndice A em até **30 páginas**. A separação entre `preambulo/`, `secoes/` e `pos_textuais/` permite manter e compilar cada capítulo de forma independente a partir de `relatorio/main.tex`.

---

## Pré-requisitos e Execução

### 1. Pré-requisitos

- **Python** $\ge 3{,}10$ com as bibliotecas listadas em `codigo/requirements.txt`:
  - `numpy>=1.24`
  - `scipy>=1.10`
  - `matplotlib>=3.7`
  - `control>=0.9.4`
- **Distribuição LaTeX** (`pdflatex` via MiKTeX ou TeX Live) com pacotes padrão (`babel`, `amsmath`, `amssymb`, `mathtools`, `booktabs`, `tikz`, `listings`, `caption`, `subcaption`, `fancyhdr`, `titlesec`, `hyperref`).

### 2. Executar as simulações numéricas

Na raiz do projeto (ou a partir de qualquer diretório), instale as dependências e execute o orquestrador:

```powershell
pip install -r codigo/requirements.txt
python codigo/main.py
```

O comando gera automaticamente `resultados/resultados_metricas.json` e os 16 arquivos de gráficos (`fig1` a `fig8` em `.pdf` e `.png`) em `relatorio/figuras/`.

### 3. Compilar o relatório técnico em PDF

Execute três passadas de `pdflatex` a partir da pasta `relatorio/` para resolver sumário, listas de ilustrações/tabelas e referências cruzadas:

```powershell
cd relatorio
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

O arquivo compilado será gravado em `relatorio/main.pdf`.

---

## Resumo dos Principais Resultados

| Controlador / Configuração | $K_p;\, K_i;\, K_d$ | $t_r$ (s) | $t_s$ (s) | $M_p$ (\%) | $e_{ss,\text{dist}}$ (ev) | $\mathrm{PM}$ ($^\circ$) | $\mu_{\max}$ (ev/s) | Atende? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P referência** ($K=0{,}369$) | $0{,}369;\; 0;\; 0$ | $4{,}216$ | $13{,}752$ | $10{,}55$ | $271{,}00$ | $57{,}97$ | $1110{,}5$ | Não |
| **PI estável** | $0{,}350;\; 0{,}030;\; 0$ | $3{,}614$ | $19{,}578$ | $34{,}67$ | $3{,}60^\dagger$ | $42{,}78$ | $1134{,}4$ | Não |
| **PI instável** (linear) | $0{,}500;\; 0{,}300;\; 0$ | N/A | N/A | N/A | N/A | $-5{,}21$ | $1752{,}5$ | Não |
| **PD saturado + D na medição** ($\tau_f=0{,}05\text{ s}$) | $1{,}280;\; 0;\; 1{,}560$ | $3{,}096$ | $4{,}762$ | $1{,}18$ | $78{,}13$ | $75{,}33$ | $1104{,}7$ | Não |
| **PID nominal linear** ($\alpha=0{,}15\text{ s}^{-1}$) | $2{,}160;\; 0{,}243;\; 2{,}900$ | $1{,}022$ | $10{,}440$ | $8{,}13$ | $0{,}15^\dagger$ | $77{,}67$ | $1397{,}0$ | Sim (linear) |
| **PID nominal sat. + BC + D-med** | $2{,}160;\; 0{,}243;\; 2{,}900$ | $2{,}802$ | $16{,}762$ | $6{,}50$ | $0{,}14^\dagger$ | $77{,}67$ | $1114{,}8$ | Não |
| **PID ajustado sat. + BC + D-med** ($\alpha=0{,}40\text{ s}^{-1}$) | $\mathbf{3{,}060;\; 0{,}648;\; 3{,}400}$ | $\mathbf{2{,}370}$ | $\mathbf{10{,}744}$ | $\mathbf{6{,}98}$ | $\mathbf{0{,}00}$ | $\mathbf{73{,}28}$ | $\mathbf{1130{,}2}$ | **Sim (Todos)** |

$^\dagger$ *Valor simulado ao término da janela de $60\text{ s}$, ainda em convergência para $100/K$ (P) ou $0{,}00\text{ eventos}$ (PI e PID nominal). BC: anti-windup por back-calculation ($T_t = 0{,}5\text{ s}$).*

O controlador **PID ajustado** ($\alpha = 0{,}40\text{ s}^{-1}$), implementado com filtro passa-baixas derivativo ($\tau_f = 0{,}05\text{ s}$), derivada aplicada exclusivamente à medição (eliminando o *derivative kick*) e *anti-windup* por *back-calculation* ($T_t = 0{,}5\text{ s}$), é o único a satisfazer todos os requisitos de projeto na presença da saturação física $[600,\,1400]\text{ eventos/s}$, recuperando-se da sobrecarga permanente em $t_{s,\text{recup}} = 5{,}950\text{ s}$ com erro estacionário nulo.
