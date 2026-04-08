##AI-Assisted. Porque para este tipo de tareas si sirve. 

import os
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import math


paletaDoble = ["#E99B16", "#1664E9"]                     
paletaTriadica = ["#A61BE4", "#E4A61B", "#1BE4A6"]   
paletaTriadicaDesempeño = ["#717473","#E99B16", "#1664E9", ]       
paletaTetradica   = ["#B2DC23", "#23DCA9", "#4D23DC", "#DC2356"]  
paletaDeBarras = paletaTriadica + paletaTetradica + paletaDoble
PaletaBase = ["#A61BE4", "#E4A61B", "#1BE4A6",   "#B2DC23", "#23DCA9", "#4D23DC", "#DC2356",  "#E99B16", "#1664E9"]


@staticmethod
def graficos_por_pipeline(pipeline: str,dfProcesado: pd.DataFrame,resultadosAux: pd.DataFrame, output_dir: str = "figuras") -> None:
    _modos_fallo_por_pipeline(dfProcesado, resultadosAux, pipeline,output_dir)
    _stacked_bar_por_metaheuristica(dfProcesado, resultadosAux, pipeline,output_dir)
    _stacked_area_por_iteracion(dfProcesado, resultadosAux, pipeline,output_dir)
    _detalle_area_por_iteracion(dfProcesado, resultadosAux, pipeline,output_dir)

@staticmethod
def graficos_globales(registros: list[dict], output_dir: str = "figuras_globales") -> None:
    _stacked_bar_global(registros,output_dir)
    _linea_por_iteracion(registros,output_dir)
    _modos_fallo_global(registros,output_dir)

@staticmethod
def _series_por_iteracion(dfProcesado: pd.DataFrame,resultadosAux: pd.DataFrame,) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (fallos, ejecuciones, optimos) indexed by Iteracion."""
    todas_iter = resultadosAux.groupby("Iteracion").size()
    por_iter   = (
        dfProcesado.groupby("Iteracion")["Solucion identica"]
        .agg(Optimos="sum", Ejecuciones=lambda s: (~s).sum())
        .reindex(todas_iter.index, fill_value=0))
    fallos = todas_iter - por_iter["Optimos"] - por_iter["Ejecuciones"]
    return fallos, por_iter["Ejecuciones"], por_iter["Optimos"]


@staticmethod
def _stacked_bar_global(registros: list[dict], output_dir) -> None:
    pipelines = [r["pipeline"]                                          for r in registros]
    optimos   = [r["dfProcesado"]["Solucion identica"].sum()            for r in registros]
    ejec      = [(~r["dfProcesado"]["Solucion identica"]).sum()         for r in registros]
    fallos    = [len(r["resultadosAux"]) - len(r["dfProcesado"])        for r in registros]

    optimos = np.array(optimos)
    ejec    = np.array(ejec)
    fallos  = np.array(fallos)
    x       = np.arange(len(pipelines))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, optimos,                      color=paletaTriadicaDesempeño[2], label="Óptimos")
    ax.bar(x, ejec,   bottom=optimos,       color=paletaTriadicaDesempeño[1], label="Ejecuciones")
    ax.bar(x, fallos, bottom=optimos + ejec, color=paletaTriadicaDesempeño[0], label="Fallos")

    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=15, ha="right")
    ax.set_ylabel("Experimentos")
    ax.set_title("Resultados globales por pipeline")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"barras_desepeño_general.pdf"),bbox_inches="tight")
    plt.close(fig)


@staticmethod
def _linea_por_iteracion(registros: list[dict], output_dir) -> None:
    fig, (ax_opt, ax_ejec) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for idx, r in enumerate(registros):
        color    = paletaDeBarras[idx % len(paletaDeBarras)]
        fallos, ejec, opts = _series_por_iteracion(
            r["dfProcesado"], r["resultadosAux"]
        )
        ax_opt.plot(opts.index,  opts.values,  marker="o", markersize=4,
                    color=color, label=r["pipeline"])
        ax_ejec.plot(ejec.index, ejec.values, marker="o", markersize=4,
                     color=color, label=r["pipeline"])

    ax_opt.set_ylabel("Óptimos")
    ax_opt.set_title("Óptimos por iteración — todos los pipelines")
    ax_opt.legend()
    ax_ejec.set_ylabel("Ejecuciones correctas")
    ax_ejec.set_xlabel("Iteración")
    ax_ejec.set_title("Ejecuciones correctas por iteración — todos los pipelines")
    ax_ejec.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"linea_ejecuciones_por_iteracion.pdf"), bbox_inches="tight")
    plt.close(fig)

@staticmethod
def _modos_fallo_global(registros: list[dict], output_dir) -> None:
    fallo_keys = sorted({key for registro in registros 
                         for key in registro["FallosTot"]
                         if key != "Total Fallos"})
    if not fallo_keys:
        return

    n_pip = len(registros)
    width = 0.8 / n_pip
    x     = np.arange(len(fallo_keys))

    fig, ax = plt.subplots(figsize=(max(7, len(fallo_keys) * 1.8), 5))
    for idx, r in enumerate(registros):
        counts = [r["FallosTot"].get(k, 0) for k in fallo_keys]
        offset = (idx - n_pip / 2 + 0.5) * width
        ax.barh(x + offset, counts, height=width,
                color=paletaDeBarras[idx % len(paletaDeBarras)],
                label=r["pipeline"])

    ax.set_yticks(x)
    ax.set_yticklabels(fallo_keys)
    ax.set_xlabel("Cantidad de fallos")
    ax.set_title("Modos de fallo por pipeline")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"barras_tipos_fallo.pdf"),bbox_inches="tight")
    plt.close(fig)

@staticmethod
def _modos_fallo_por_pipeline(dfProcesado, resultadosAux, pipeline, output_dir):
    fallos_aux = resultadosAux[~resultadosAux.index.isin(dfProcesado.index)]
    if "TipoFallo" not in fallos_aux.columns or fallos_aux.empty:
        return
    pivot = (
        fallos_aux.groupby(["TipoFallo", "Metaheuristica"])
        .size()
        .unstack(fill_value=0)
    )
    metaheuristicas = pivot.columns.tolist()
    fallo_keys      = pivot.index.tolist()
    n_mh  = len(metaheuristicas)
    width = 0.8 / n_mh
    x     = np.arange(len(fallo_keys))

    fig, ax = plt.subplots(figsize=(max(7, len(fallo_keys) * 1.8), 5))
    for idx, mh in enumerate(metaheuristicas):
        counts = pivot[mh].values
        offset = (idx - n_mh / 2 + 0.5) * width
        ax.barh(x + offset, counts, height=width,
                color=paletaDeBarras[idx % len(paletaDeBarras)],
                label=mh)

    ax.xaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_yticks(x)
    ax.set_yticklabels(fallo_keys)
    ax.set_xlabel("Cantidad de fallos")
    ax.set_title(f"Tipos de fallo por metaheurística — {pipeline}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(metaheuristicas), frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, f"{pipeline}_barras_tipos_fallo.pdf"), bbox_inches="tight")
    plt.close(fig)

@staticmethod
def _stacked_bar_por_metaheuristica(dfProcesado, resultadosAux, pipeline, output_dir):
    totales = resultadosAux.groupby("Metaheuristica").size()
    por_mh  = dfProcesado.groupby("Metaheuristica")["Solucion identica"].agg(
        Optimos     = "sum",
        Ejecuciones = lambda s: (~s).sum(),
    )
    df = totales.rename("Total").to_frame().join(por_mh, how="left").fillna(0)
    df["Fallos"] = df["Total"] - df["Optimos"] - df["Ejecuciones"]
    df_pct = df[["Optimos", "Ejecuciones", "Fallos"]].div(df["Total"], axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.8)))
    y   = np.arange(len(df))
    bot = np.zeros(len(df))
    labels = ["Óptimos", "Ejecuciones", "Fallos"]
    for col, color, label in [
        ("Optimos",     paletaTriadicaDesempeño[2], "Óptimos"),
        ("Ejecuciones", paletaTriadicaDesempeño[1], "Ejecuciones"),
        ("Fallos",      paletaTriadicaDesempeño[0], "Fallos"),
    ]:
        vals = df_pct[col].values
        ax.barh(y, vals, left=bot, color=color, label=label)
        bot += vals

    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Proporción")
    ax.set_title(f"Comportamiento por metaheurística — {pipeline}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, f"{pipeline}_barras_desepeño.pdf"), bbox_inches="tight")
    plt.close(fig)

@staticmethod
def _stacked_area_por_iteracion(dfProcesado, resultadosAux, pipeline, output_dir):
    fallos, ejec, opts = _series_por_iteracion(dfProcesado, resultadosAux)
    total = fallos + ejec + opts
    fallos_pct = fallos / total * 100
    ejec_pct   = ejec   / total * 100
    opts_pct   = opts   / total * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        opts.index,
        opts_pct, ejec_pct, fallos_pct,
        labels=["Óptimos", "Ejecuciones", "Fallos"],
        colors=paletaTriadica[::-1],
        alpha=0.85,
    )
    ax.set_ylim(0, 100)
    ax.set_xlim(opts.index[0], opts.index[-1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Proporción")
    ax.set_title(f"Resultados por iteración — {pipeline}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(output_dir, f"{pipeline}_area_resumen.pdf"), bbox_inches="tight")
    plt.close(fig)

@staticmethod
def _detalle_area_por_iteracion(dfProcesado: pd.DataFrame,resultadosAux: pd.DataFrame,pipeline: str, output_dir) -> None:
    fallos, ejec, opts = _series_por_iteracion(dfProcesado, resultadosAux)
    idx = opts.index

    fig, (ax_opt, ax_ejec, ax_fallo) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, series, color, label in [
        (ax_opt,   opts,   paletaTriadicaDesempeño[2], "Óptimos"),
        (ax_ejec,  ejec,   paletaTriadicaDesempeño[1], "Ejecuciones"),
        (ax_fallo, fallos, paletaTriadicaDesempeño[0], "Fallos"),
    ]:
        ax.fill_between(idx, series.values, alpha=0.85, color=color)
        ax.plot(idx, series.values, color=color)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.set_xlim(opts.index[0], opts.index[-1])
        ax_fallo.set_xlabel("Iteración")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    fig.suptitle(f"Detalle resultados por iteración — {pipeline}")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pipeline}_area_detalle.pdf"),bbox_inches="tight")
    plt.close(fig)





