from collections import Counter
import math
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

# =========================================================
# Funções para Assimetria (Skewness) e Curtose (Excesso)
# =========================================================
def _to_1d_array(x):
    a = np.asarray(x, dtype=float).ravel()
    return a[~np.isnan(a)]

def skewness(x):
    """Assimetria amostral (Fisher-Pearson ajustada)."""
    a = _to_1d_array(x)
    n = a.size
    if n < 3:
        return 0.0
    m = a.mean()
    m2 = np.mean((a - m) ** 2)
    m3 = np.mean((a - m) ** 3)
    if m2 <= 0:
        return 0.0
    g1 = m3 / (m2 ** 1.5)
    return float((n * math.sqrt(n - 1)) / (n - 2) * g1)

def kurtosis_excess(x):
    """Excesso de curtose amostral (Fisher) — Normal => 0."""
    a = _to_1d_array(x)
    n = a.size
    if n < 4:
        return 0.0
    m = a.mean()
    m2 = np.mean((a - m) ** 2)
    m4 = np.mean((a - m) ** 4)
    if m2 <= 0:
        return 0.0
    term = ((n + 1) * (m4 / (m2 ** 2))) - (3 * (n - 1))
    g2_adj = ((n - 1) / ((n - 2) * (n - 3))) * term
    return float(g2_adj)

# =========================================================
# Dados (atos praticados) e períodos
# =========================================================
dados = [
    48879, 60946, 30532, 29436, 32699, 29897, 31407, 32577,
    33981, 34406, 38188, 41169, 52124, 54673, 42255, 52969,
    54576, 54040, 52570, 60940, 56473, 52106, 52622, 50310,
    51019, 51609, 45854, 51343, 33144, 45215, 38576, 40718,
    37392, 46629, 41810, 37205, 29192, 37924, 34933
]

periodos = [
    "2005", "2006", "2007-1", "2007-2", "2008-1", "2008-2", "2009-1", "2009-2",
    "2010-1", "2010-2", "2011-1", "2011-2", "2012-1", "2012-2", "2013-1", "2013-2",
    "2014-1", "2014-2", "2015-1", "2015-2", "2016-1", "2016-2", "2017-1", "2017-2",
    "2018-1", "2018-2", "2019-1", "2019-2", "2020-1", "2020-2", "2021-1", "2021-2",
    "2022-1", "2022-2", "2023-1", "2023-2", "2024-1", "2024-2", "2025-1"
]

# =========================================================
# Estatísticas
# =========================================================
media = np.mean(dados)
mediana = np.median(dados)

try:
    modas = st.multimode(dados)
except Exception:
    cont = Counter(dados)
    maxf = max(cont.values())
    modas = [v for v, f in cont.items() if f == maxf]
amodal = len(modas) == len(dados)

desvio_padrao = np.std(dados, ddof=1)  # amostral
cv = (desvio_padrao / media) * 100 if media else 0.0

# novas métricas
assimetria = skewness(dados)
curtose_excesso = kurtosis_excess(dados)

# =========================================================
# Impressão no console
# =========================================================
print("\n📊 Estatísticas do Cartório")
print(f"Média: {media:,.2f} atos")
print(f"Mediana: {mediana:,} atos")
if amodal:
    print("Moda: amodal (não há valores que se repetem)")
else:
    print(f"Moda(s): {', '.join(str(m) for m in modas)} atos")
print(f"Desvio Padrão (amostral): {desvio_padrao:,.2f} atos")
print(f"Coeficiente de Variação: {cv:,.2f}%")
print(f"Assimetria (g₁): {assimetria:,.4f}")
print(f"Curtose (excesso): {curtose_excesso:,.4f}  (Normal=0)")

# =========================================================
# Gráfico de colunas
# =========================================================
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.bar(range(len(dados)), dados, edgecolor="black")
ax1.set_title("Atos praticados por período")
ax1.set_xlabel("Período")
ax1.set_ylabel("Quantidade de atos")
ax1.set_xticks(range(len(periodos)))
ax1.set_xticklabels(periodos, rotation=45, ha="right")
fig1.tight_layout()
fig1.savefig("grafico_barras.png", dpi=300)
plt.show()

# =========================================================
# Histograma
# =========================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.hist(dados, bins=10, edgecolor="black")
ax2.set_title("Distribuição dos Atos")
ax2.set_xlabel("Quantidade de atos")
ax2.set_ylabel("Frequência")
fig2.tight_layout()
fig2.savefig("histograma_distribuicao.png", dpi=300)
plt.show()

# =========================================================
# Gráfico de Onda (Linha Temporal) com Assimetria + extremos
# =========================================================
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(periodos, dados, marker="o", linestyle="-", label="Atos praticados")

ax3.annotate(
    f"Assimetria (g₁) = {assimetria:.3f}\nLeve cauda à direita",
    xy=(0.01, 0.95), xycoords="axes fraction",
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8)
)

max_idx = np.argmax(dados)
min_idx = np.argmin(dados)
ax3.scatter(periodos[max_idx], dados[max_idx], color="red", s=100, zorder=5, label="Máximo")
ax3.scatter(periodos[min_idx], dados[min_idx], color="green", s=100, zorder=5, label="Mínimo")
ax3.text(max_idx, dados[max_idx] + 600, f"{periodos[max_idx]}: {dados[max_idx]:,}", color="red", ha="center")
ax3.text(min_idx, dados[min_idx] - 1200, f"{periodos[min_idx]}: {dados[min_idx]:,}", color="green", ha="center")

ax3.set_title("Atos praticados por período (Gráfico de Onda)")
ax3.set_xlabel("Período")
ax3.set_ylabel("Quantidade de atos")
ax3.set_xticks(range(len(periodos)))
ax3.set_xticklabels(periodos, rotation=45, ha="right")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend()
fig3.tight_layout()
fig3.savefig("grafico_onda_assimetria.png", dpi=300)
plt.show()

# =========================================================
# Curvas teóricas no formato “suave de sino”
# (Assimetria Negativa, Simétrica e Positiva) — compatível Py 3.13
# =========================================================
def phi(x):
    x = np.asarray(x, dtype=float)
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)

# Vetoriza math.erf para aceitar arrays (sem np.erf)
_erf_ufunc = np.frompyfunc(math.erf, 1, 1)
def Phi(x):
    z = np.asarray(x, dtype=float) / np.sqrt(2.0)
    return 0.5 * (1.0 + np.array(_erf_ufunc(z), dtype=float))

def skew_normal_pdf(x, alpha):
    x = np.asarray(x, dtype=float)
    return 2.0 * phi(x) * Phi(alpha * x)

x = np.linspace(-4, 4, 600)
fig4, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)

curvas = [
    {"alpha": -6, "titulo": "Assimetria Negativa"},
    {"alpha":  0, "titulo": "Simétrica"},
    {"alpha":  6, "titulo": "Assimetria Positiva"},
]

for ax, cfg in zip(axes, curvas):
    y = skew_normal_pdf(x, cfg["alpha"])
    ax.plot(x, y, linewidth=2)
    ax.axhline(0, color="black", linewidth=2)  # base
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(cfg["titulo"])
    ax.set_xlim(-4, 4); ax.set_ylim(0, y.max() * 1.15)

fig4.tight_layout()
fig4.savefig("curvas_assimetria.png", dpi=300)
plt.show()

# Rodar:
# python main.py
