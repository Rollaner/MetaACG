##AI-Assisted. Porque para este tipo de tareas si sirve. 

import os
import numpy as np
import matplotlib
import matplotlib.patheffects
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import math


paletaDoble = ["#E99B16", "#1664E9"]                     
paletaTriadica = ["#A61BE4", "#E4A61B", "#1BE4A6"]         
paletaTetradica   = ["#B2DC23", "#23DCA9", "#4D23DC", "#DC2356"]  
paletaDeBarras = paletaTriadica + paletaTetradica + paletaDoble

def graficos_por_pipeline(Fallos, desempeñoPorSolver, metricasRendimiento,totalExperimentos,
                          pipeline: str,
                          output_dir: str = "figuras"):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Pie: Tasa de fallo global (fallos vs éxitos)
    # ------------------------------------------------------------------
    total_fallos = Fallos.get("Total Fallos", 0)
    total_exitos = Fallos.get("Total Exitos", 0)
    total = total_fallos + total_exitos
    if total > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        valores = np.array([total_fallos, total_exitos], dtype=float)
        etiquetas_base = ["Fallos", "Éxitos"]
        colores = paletaDoble

        fig, ax = plt.subplots(figsize=(7, 6))

        wedges = ax.pie(
            valores,
            labels=None,
            autopct=None,
            startangle=90,
            colors=colores
        )

        porcentajes = valores / valores.sum() * 100
        legend_labels = [
            f"{name} ({pct:.1f}%)"
            for name, pct in zip(etiquetas_base, porcentajes)
        ]

        ax.legend(
            wedges[0],
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(len(legend_labels), 4)
        )

        ax.set_title("lo que sea...")
        ax.axis("equal")
        plt.tight_layout()
        ax.set_title(f"Tasa de fallo global ({pipeline})")
        ax.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pipeline}_pie_tasa_fallo.pdf"))
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2) Barras apiladas: desempeño por solver (éxitos vs fallos)
    # ------------------------------------------------------------------
    if not desempeñoPorSolver.empty:
        labels = desempeñoPorSolver["Metaheuristica"].tolist()
        exitos = desempeñoPorSolver["TotalExitos"].to_numpy().astype(float)
        fallosPorSolver = desempeñoPorSolver["TotalFallos"].to_numpy().astype(float)
        fallos = totalExperimentos - exitos
        x = np.arange(len(labels))
        coloresExito = [paletaTriadica[i % len(paletaTriadica)]
                         for i in range(len(labels))]
        colorFallo = "#BBBBBB"   

        fig, ax = plt.subplots(figsize=(7, 4))

        bars_success = ax.bar(
            x,
            exitos,
            edgecolor="black",
            linewidth=1.0,
            color=coloresExito,
            width=0.7,
            label="Éxitos",
        )

        bars_fail = ax.bar(
            x,
            fallos,
            bottom=exitos,
            edgecolor="black",
            linewidth=1.0,
            color=colorFallo,
            width=0.7,
            label="Fallos",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Número de ejecuciones")
        ax.set_title(f"Desempeño por solver (éxitos vs fallos) ({pipeline})")

        max_val = np.max(totalExperimentos)
        configurar_grid_barras(ax, max_val, es_proporcion=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            tasa_exito = np.where(totalExperimentos > 0, exitos / totalExperimentos * 100, 0.0)
        legend_success_items = [
            f"{name}: {int(e)} éxitos ({rate:.1f}%)"
            for name, e, rate in zip(labels, exitos, tasa_exito)
        ]

        total_fallos = int(fallosPorSolver.sum())
        total_porcentaje_fallos = fallosPorSolver.sum() / totalExperimentos * 100
        legend_fail_item = f"Fallos totales: {total_fallos} ({total_porcentaje_fallos:.1f}%)"
        from matplotlib.patches import Patch
        legend_handles = []
        for i, solver in enumerate(labels):
            legend_handles.append(Patch(
                facecolor=coloresExito[i],
                edgecolor="black",
                label=legend_success_items[i]
            ))
        legend_handles.append(Patch(
            facecolor=colorFallo,
            edgecolor="black",
            label=legend_fail_item
        ))
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=1,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir,
            f"{pipeline}_barras_exitos_por_solver.pdf"
        ), bbox_inches="tight")

        plt.close(fig)


    # ------------------------------------------------------------------
    # 3) Barras: tipos de fallo (sin incluir Totales)
    # ------------------------------------------------------------------
    tipos = []
    cantidades = []
    for key, count in Fallos.items():
        if key not in ["Total Fallos", "Total Exitos"] and count > 0:
            tipos.append(key.replace("_", " "))
            cantidades.append(count)

    if len(cantidades) > 0:
        valores = np.array(cantidades, dtype=float)
        colores = [paletaTetradica[i % len(paletaTetradica)]
                   for i in range(len(tipos))]

        fig, ax = plt.subplots(figsize=(7, 4))
        y = np.arange(len(tipos))
        bars = []
        for i, val in enumerate(valores):
            rect = ax.barh(
                y[i],
                val,
                edgecolor="black",
                linewidth=1.5,
                color=colores[i],
                height=0.7,
            )[0]
            bars.append(rect)

        ax.set_yticks(y)
        ax.set_yticklabels(tipos)
        ax.invert_yaxis()  
        ax.set_xlabel("Cantidad de fallos")
        ax.set_title(f"Tipos de fallos ({pipeline})")

        max_val = np.max(valores)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.4)

        porcentajes = valores / valores.sum() * 100
        legend_labels = [
            f"{name} ({int(val)} fallos, {pct:.1f}%)"
            for name, val, pct in zip(tipos, valores, porcentajes)
        ]

        ax.legend(
            bars,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(legend_labels), 3),
        )

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir,
            f"{pipeline}_pie_tipos_fallo.pdf"
        ), bbox_inches="tight")

        plt.close(fig)


        # ------------------------------------------------------------------
        # 4) y 5) Whisker plots: error porcentual y error absoluto + IC
        # ------------------------------------------------------------------
    if not metricasRendimiento.empty:
        x = np.arange(len(metricasRendimiento))
        labels = metricasRendimiento["Metaheuristica"].tolist()

        means_pct = metricasRendimiento["Promedio_Error_Porcentual"].to_numpy()
        ic_min_pct = metricasRendimiento["Intervalo_de_Confianza_Error_Porcentual_Minimo"].to_numpy()
        ic_max_pct = metricasRendimiento["Intervalo_de_Confianza_Error_Porcentual_Maximo"].to_numpy()

        err_lower = means_pct - ic_min_pct
        err_upper = ic_max_pct - means_pct

        fig, ax = plt.subplots(figsize=(7, 4))

        # Whisker plot
        for i, m in enumerate(labels):
            ax.errorbar(
                x[i],
                means_pct[i],
                yerr=[[err_lower[i]], [err_upper[i]]],
                fmt="o",
                markersize=6,
                color=paletaDeBarras[i % len(paletaDeBarras)],
                ecolor=paletaDeBarras[i % len(paletaDeBarras)],
                elinewidth=1.5,
                capsize=5
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Error porcentual medio (%)")
        ax.set_title(f"Error porcentual por metaheurística ({pipeline})")
        ax.axhline(0, color="black", linewidth=1.0)

        ymin = min(np.min(ic_min_pct), 0)
        ymax = max(np.max(ic_max_pct), 0)
        margin = 0.05 * (ymax - ymin) if ymax != ymin else 1.0
        ax.set_ylim(ymin - margin, ymax + margin)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

        handles = [
            Patch(facecolor=paletaDeBarras[i % len(paletaDeBarras)], edgecolor="black")
            for i in range(len(labels))
        ]
        legendaPorBarra(
            ax,
            handles,
            labels,
            means_pct,
            fmt="{:.2f}",
            title="Error porcentual medio"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pipeline}_barras_error_porcentual.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    if not metricasRendimiento.empty:
            x = np.arange(len(metricasRendimiento))
            labels = metricasRendimiento["Metaheuristica"].tolist()

            # ---- porcentual ----
            means_pct = metricasRendimiento["Promedio_Error_Porcentual"].to_numpy()
            ic_min_pct = metricasRendimiento["Intervalo_de_Confianza_Error_Porcentual_Minimo"].to_numpy()
            ic_max_pct = metricasRendimiento["Intervalo_de_Confianza_Error_Porcentual_Maximo"].to_numpy()

            plot_whisker_ci(
                x=x,
                labels=labels,
                means=means_pct,
                ic_min=ic_min_pct,
                ic_max=ic_max_pct,
                palette=paletaDeBarras,
                ylabel="Error porcentual medio (%)",
                title=f"Error porcentual por metaheurística ({pipeline})",
                legend_title="Error porcentual medio",
                output_path=os.path.join(
                    output_dir,
                    f"{pipeline}_barras_error_porcentual.pdf",
                ),
                fmt="{:.2f}",
                force_zero=False,
            )

            # ---- absoluto ----
            means_abs = metricasRendimiento["Promedio_Error_Abs"].to_numpy()
            ic_min_abs = metricasRendimiento["Intervalo_de_Confianza_Error_Absoluto_Minimo"].to_numpy()
            ic_max_abs = metricasRendimiento["Intervalo_de_Confianza_Error_Absoluto_Maximo"].to_numpy()

            plot_whisker_ci(
                x=x,
                labels=labels,
                means=means_abs,
                ic_min=ic_min_abs,
                ic_max=ic_max_abs,
                palette=paletaDeBarras,
                ylabel="Error absoluto medio",
                title=f"Error absoluto por metaheurística ({pipeline})",
                legend_title="Error absoluto medio",
                output_path=os.path.join(
                    output_dir,
                    f"{pipeline}_barras_error_absoluto.pdf",
                ),
                fmt="{:.2f}",
                force_zero=True,  
            )

def graficos_globales(comparacion, fisher_df,
                      output_dir: str = "figuras_globales"):
    os.makedirs(output_dir, exist_ok=True)


    PaletaBase = [
        "#A61BE4", "#E4A61B", "#1BE4A6",   
        "#B2DC23", "#23DCA9", "#4D23DC", "#DC2356",  
        "#E99B16", "#1664E9"              
    ]

    metas_comp = comparacion["Metaheuristica"].unique().tolist() if not comparacion.empty else []
    metas_fish = fisher_df["Metaheuristica"].unique().tolist() if not fisher_df.empty else []
    metaheuristicas = list(dict.fromkeys(metas_comp + metas_fish)) 

    if len(metaheuristicas) == 0:
        return

    color_meta = {
        m: PaletaBase[i % len(PaletaBase)]
        for i, m in enumerate(metaheuristicas)
    }

    # --------------------------------------------------------------
    # 1) Comparación de pipelines con IC: Delta error porcentual y absoluto
    #    → whisker plots (punto + intervalo)
    # --------------------------------------------------------------
    if not comparacion.empty:
        dfc = comparacion.copy()
        labels = dfc["Metaheuristica"].tolist()
        x = np.arange(len(dfc))

        # ---------- ΔError porcentual ----------
        means_pct    = dfc["Delta_Error_Pct"].to_numpy()
        ic_min_pct   = dfc["IC_Delta_Error_Pct_Menor"].to_numpy()
        ic_max_pct   = dfc["IC_Delta_Error_Pct_Maximo"].to_numpy()

        # diferencias asimétricas para errorbar
        yerr_lower_pct = means_pct - ic_min_pct
        yerr_upper_pct = ic_max_pct - means_pct
        yerr_pct = np.vstack([yerr_lower_pct, yerr_upper_pct])

        fig, ax = plt.subplots(figsize=(7, 4))

        # Whisker plot: puntos + IC
        for i, m in enumerate(labels):
            ax.errorbar(
                x[i],
                means_pct[i],
                yerr=[[yerr_lower_pct[i]], [yerr_upper_pct[i]]],
                fmt="o",              # marcador
                markersize=6,
                color=color_meta[m],  # color de metaheurística
                ecolor=color_meta[m], # mismo color para el whisker
                elinewidth=1.5,
                capsize=5
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(r"$\Delta$ Error porcentual (CD - SD) [%]")
        ax.set_title("Comparación de pipelines: ΔError porcentual por metaheurística")
        ax.axhline(0, color="black", linewidth=1.0)

        min_ci_pct = np.min(ic_min_pct)
        max_ci_pct = np.max(ic_max_pct)

        ymin = min(min_ci_pct, 0)
        ymax = max(max_ci_pct, 0)

        if ymax == ymin:
            margin = 1.0
        else:
            margin = 0.05 * (ymax - ymin)

        ax.set_ylim(ymin - margin, ymax + margin)

        # Ticks decentes (negativos y positivos) + grid
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

        # Legenda: patches dummy para que leyendaPorBarra siga funcionando igual
        handles = [
            Patch(facecolor=color_meta[m], edgecolor="black")
            for m in labels
        ]

        legendaPorBarra(
            ax,
            handles,
            labels,
            means_pct,
            fmt="{:.2f}",
            title=r"$\Delta$ Error porcentual"
        )

        plt.tight_layout()
        fig_path_pct = os.path.join(output_dir, "comparacion_delta_error_porcentual.pdf")
        plt.savefig(fig_path_pct, bbox_inches="tight")
        plt.close(fig)

        means_abs   = dfc["Delta_Error_Abs"].to_numpy()
        ic_min_abs  = dfc["IC_Delta_Error_Abs_Menor"].to_numpy()
        ic_max_abs  = dfc["IC_Delta_Error_Abs_Maximo"].to_numpy()

        yerr_lower_abs = means_abs - ic_min_abs
        yerr_upper_abs = ic_max_abs - means_abs
        yerr_abs = np.vstack([yerr_lower_abs, yerr_upper_abs])

        fig, ax = plt.subplots(figsize=(7, 4))

        for i, m in enumerate(labels):
            ax.errorbar(
                x[i],
                means_abs[i],
                yerr=[[yerr_lower_abs[i]], [yerr_upper_abs[i]]],
                fmt="o",
                markersize=6,
                color=color_meta[m],
                ecolor=color_meta[m],
                elinewidth=1.5,
                capsize=5
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(r"$\Delta$ Error absoluto (CD - SD)")
        ax.set_title("Comparación de pipelines: ΔError absoluto por metaheurística")
        ax.axhline(0, color="black", linewidth=1.0)

        min_ci_abs = np.min(ic_min_abs)
        max_ci_abs = np.max(ic_max_abs)

        ymin = min(min_ci_abs, 0)
        ymax = max(max_ci_abs, 0)

        if ymax == ymin:
            margin = 1.0
        else:
            margin = 0.05 * (ymax - ymin)

        ax.set_ylim(ymin - margin, ymax + margin)

        # Ticks bien repartidos (negativos) + grid
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

        handles = [
            Patch(facecolor=color_meta[m], edgecolor="black")
            for m in labels
        ]

        legendaPorBarra(
            ax,
            handles,
            labels,
            means_abs,
            fmt="{:.2f}",
            title=r"$\Delta$ Error absoluto"
        )

        plt.tight_layout()
        fig_path_abs = os.path.join(output_dir, "comparacion_delta_error_absoluto.pdf")
        plt.savefig(fig_path_abs, bbox_inches="tight")
        plt.close(fig)
    # --------------------------------------------------------------
    # 2) Fisher: p-values
    # --------------------------------------------------------------
    if not fisher_df.empty:
        dff = fisher_df.copy()
        labels = dff["Metaheuristica"].tolist()
        x = np.arange(len(dff))

        # p-valor bilateral
        p2 = dff["PValue_Bilateral"].to_numpy()

        fig, ax = plt.subplots(figsize=(7, 4))
        handles = []
        for i, (m, val) in enumerate(zip(labels, p2)):
            rect = ax.bar(
                x[i],
                val,
                edgecolor="black",
                linewidth=1.5,
                color=color_meta[m],
                width=0.7
            )[0]
            handles.append(rect)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("p-valor bilateral")
        ax.set_title("Prueba de Fisher: p-valor bilateral por metaheurística")
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)

        max_val = np.max(p2)
        configurar_grid_barras(ax, max_val, es_proporcion=True)

        legendaPorBarra(
            ax,
            handles,
            labels,
            p2,
            fmt="{:.3f}",
            title="p-valor bilateral"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 "fisher_pvalue_bilateral.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

        # p-valor FP mejor
        p1 = dff["PValue_FPMejor"].to_numpy()

        fig, ax = plt.subplots(figsize=(7, 4))
        handles = []
        for i, (m, val) in enumerate(zip(labels, p1)):
            rect = ax.bar(
                x[i],
                val,
                edgecolor="black",
                linewidth=1.5,
                color=color_meta[m],
                width=0.7
            )[0]
            handles.append(rect)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("p-valor (FP mejor que Control)")
        ax.set_title("Prueba de Fisher: p-valor unilateral por metaheurística")
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)

        max_val = np.max(p1)
        configurar_grid_barras(ax, max_val, es_proporcion=True)

        legendaPorBarra(
            ax,
            handles,
            labels,
            p1,
            fmt="{:.3f}",
            title="p-valor FP mejor"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 "fisher_pvalue_fpmejor.pdf"),
                    bbox_inches="tight")
        plt.close(fig)


def plot_whisker_ci(
    x,
    labels,
    means,
    ic_min,
    ic_max,
    palette,
    ylabel,
    title,
    legend_title,
    output_path,
    fmt="{:.2f}",
    force_zero=False,
):
    """
    Whisker plot (punto + IC asimétrico) para una métrica cualquiera.
    """
    means = np.asarray(means)
    ic_min = np.asarray(ic_min)
    ic_max = np.asarray(ic_max)

    err_lower = means - ic_min
    err_upper = ic_max - means

    fig, ax = plt.subplots(figsize=(7, 4))

    # puntos + IC
    for i, _ in enumerate(labels):
        ax.errorbar(
            x[i],
            means[i],
            yerr=[[err_lower[i]], [err_upper[i]]],
            fmt="o",
            markersize=6,
            color=palette[i % len(palette)],
            ecolor=palette[i % len(palette)],
            elinewidth=1.5,
            capsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if force_zero:
        ax.axhline(0, color="black", linewidth=1.0)

    # ---- límites del eje Y ----
    ymin = np.min(ic_min)
    ymax = np.max(ic_max)

    if force_zero:
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)

    # Para el caso de error porcentual (force_zero=False), esto hace
    # que el eje arranque cerca del IC mínimo (ej. ~100%) y no en 0.
    if ymax == ymin:
        margin = 1.0
    else:
        margin = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - margin, ymax + margin)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

    # leyenda con patches dummy (como antes)
    handles = [
        Patch(facecolor=palette[i % len(palette)], edgecolor="black")
        for i in range(len(labels))
    ]

    legendaPorBarra(
        ax,
        handles,
        labels,
        means,
        fmt=fmt,
        title=legend_title,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def configurar_grid_barras(ax, max_val, es_proporcion=False):
    """
    Pone una grilla horizontal razonable, con como mucho ~10 ticks.
    No revienta aunque max_val sea enorme o NaN.
    """
    ax.set_axisbelow(True)

    # Manejo de NaN / inf / <= 0
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0

    if es_proporcion:
        # cosas tipo p-values o tasas (0–1)
        top = max(1.0, float(max_val))
        num_ticks = 10
        ticks = np.linspace(0.0, top, num_ticks + 1)
    else:
        top = float(max_val)
        # orden de magnitud (10^n)
        magnitude = 10 ** math.floor(math.log10(top))
        step = magnitude

        # Que no haya más de ~10 ticks
        num_steps = math.ceil(top / step)
        while num_steps > 10:
            step *= 2
            num_steps = math.ceil(top / step)

        top = step * num_steps
        ticks = np.arange(0.0, top + step * 0.5, step)

    ax.set_yticks(ticks)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

def legendaPorBarra(ax, bars, labels, values, fmt="{:.2f}", title=None):
    """
    Crea una leyenda donde cada entrada es: 'label (valor formateado)'.

    - bars: lista de objetos Rectangle devueltos por ax.bar(...)[0]
    - labels: nombres de cada barra
    - values: valores numéricos asociados
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    legend_labels = [
        f"{lbl} ({fmt.format(val)})"
        for lbl, val in zip(labels, values)
    ]
    ax.legend(
        bars,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(legend_labels), 4),
        title=title
    )

def graficarMejoraPorIteracion(tablas: dict[str, any], pipeline: str, output_dir: str = "figuras_por_iteracion"):
    os.makedirs(output_dir, exist_ok=True)
    colores = {
        'Estable_OK':  paletaTetradica[0],
        'Mejora':      paletaTetradica[1],
        'Regresion':   paletaTetradica[2],
        'Estable_Mal': paletaTetradica[3],
    }
    etiquetas = {
        'Estable_OK':  'Estable (óptimo)',
        'Mejora':      'Mejora',
        'Regresion':   'Regresión',
        'Estable_Mal': 'Estable (sin óptimo)',
    }
    categorias = list(colores.keys())

    for nombre, df in tablas.items():
        if df.empty:
            continue

        x = np.arange(len(df))
        labels = df['Transicion'].tolist()

        fig, ax = plt.subplots(figsize=(8, 5))

        # 1. Prepare data for stacking
        # We extract the categories in order
        y_values = [df[cat].to_numpy().astype(float) for cat in categorias]
        
        # 2. Create the Stacked Area Plot
        # baseline='zero' makes it easy to read as a trend
        ax.stackplot(
            x, y_values,
            labels=[etiquetas[cat] for cat in categorias],
            colors=[colores[cat] for cat in categorias],
            alpha=0.8
        )

        # 3. Add Labels in the center of the bands
        # We calculate the cumulative sum to find the vertical center of each band
        cumulative_y = np.cumsum(y_values, axis=0)
        for i, cat in enumerate(categorias):
            current_vals = y_values[i]
            bottoms = cumulative_y[i-1] if i > 0 else np.zeros(len(x))
            
            for xi, val, bottom in zip(x, current_vals, bottoms):
                if val > 0:  # Changed to > 0 to see all data
                    yi = bottom + (val / 2)
                    
                    # LOGIC FIX: Adjust alignment based on X position to avoid Y-axis overlap
                    ha = 'center'
                    if xi == 0:
                        ha = 'left'   # Push away from Y-axis
                    elif xi == len(x) - 1:
                        ha = 'right'  # Push away from right edge
                    
                    ax.annotate(
                        f"{int(val)}",
                        xy=(xi, yi),
                        xytext=(0, 0), # Kept at 0 to stay exactly on the marker line
                        textcoords="offset points",
                        ha=ha,
                        va='center',
                        fontsize=8,
                        color='white',
                        fontweight='bold',
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground='black')]
                    )

        # 4. Formatting Fixes
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        
        # ADDED MARGINS: This prevents the dots/labels from touching the axes
        ax.set_xmargin(0.1) 
        
        # Ensure the Y axis starts at 0 and has room at the top
        ax.set_ylim(0, df[categorias].sum(axis=1).max() * 1.1)

        # 4. Formatting
        ax.set_ylabel("Número de problemas")
        ax.set_title(f"Evolución de Estados — {nombre} ({pipeline})")
        
        # Set limits to tighten the view
        ax.set_xlim(0, len(df)-1)
        ax.set_ylim(0, df[categorias].sum(axis=1).max() * 1.05)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{pipeline}_area_evolucion_{nombre}.pdf"),
            bbox_inches="tight"
        )
        plt.close(fig)

       

