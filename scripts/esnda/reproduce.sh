#!/usr/bin/env bash
# Reproduce the washout-free ESN-DA paper: every run, then every figure, in the order of the paper.
#
#   ./reproduce.sh [synchronization] [parameter] [localization]      (default: all, from scripts/esnda)
#
# A run is skipped when every one of its outputs exists (FORCE=1 reruns them all), so a second call only draws the
# figures; DRY_RUN=1 lists the steps that would run, and runs nothing. The figures go to $PAPER (default: figs/ here). N_WORKERS (default 2) sets the
# workers of the pools: a worker of L96-10s holds about 7 GB, one of L96-40 tens of GB. The full set of runs takes
# days on a workstation; the trained networks are kept in the store of romda (results/esn_configs/), so a network is
# trained once. The m = 1000 reference of fig:localized_covariance (row e, about 41 GB) runs only with FULL=1.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/../../src${PYTHONPATH:+:$PYTHONPATH}"   # romda of this checkout
export PAPER="${PAPER:-$PWD/figs}" MPLBACKEND=Agg N_WORKERS="${N_WORKERS:-2}"
mkdir -p "$PAPER"
PY="${PYTHON:-python}"

# step OUTPUT... -- [VAR=value ...] script [args]: run the script unless every output exists
step() {
    local outs=()
    while [ "$1" != "--" ]; do outs+=("$1"); shift; done
    shift
    if [ -z "${FORCE:-}" ] && [ ${#outs[@]} -gt 0 ]; then
        local o all=1
        for o in "${outs[@]}"; do [ -e "$o" ] || all=0; done
        if [ "$all" = 1 ]; then echo "skip (outputs exist): $*"; return; fi
    fi
    local vars=()
    while [[ "$1" == *=* ]]; do vars+=("$1"); shift; done
    echo "+ ${vars[*]:-} $*"
    if [ -n "${DRY_RUN:-}" ]; then return; fi
    env ${vars[@]+"${vars[@]}"} "$PY" "$@"
}
figure() { step -- "$@"; }

synchronization() {   # Sect. 5.1: fig:l63_forecast, the partial observations, fig:sync; Appendix A, L63
    local m s
    for m in 5 10 20 30; do
        step results/partial/partial_table_lorenz63_m$m.json -- M=$m run/run_partial_table.py 100
        step results/partial/partial_table_lorenz63_m${m}_z_noreject.json -- M=$m SUBSETS=2 TAG=_z_noreject run/run_partial_table.py 100
        step results/partial/partial_table_lorenz63_m${m}_mixture2_z.json -- M=$m SUBSETS=2 TAG=_z MIXTURE=2 run/run_partial_table.py 100
    done
    step results/partial/lorenz63_phase.npz -- run/run_partial_table.py phase
    figure plot/plot_lorenz63_phase.py
    local S=results/synchronization/synchronization
    step ${S}_lorenz63.npz -- CDD_STD=0.2 run/run_synchronization.py lorenz63
    step ${S}_lorenz63_s2.npz -- CDD_STD=0.2 SENSORS=2 run/run_synchronization.py lorenz63
    step ${S}_lorenz63_s3_2_1_a12.npz -- CDD_STD=0.2 SENSORS=3,2,1 N_ANALYSES=1 TAG=_a12 run/run_synchronization.py lorenz63
    step ${S}_lorenz63_s3_cddnoise.npz -- CDD_STD=0.05 SENSORS=3 TAG=_cddnoise run/run_synchronization.py lorenz63
    step ${S}_lorenz63_s2_1_cddnoise.npz -- CDD_STD=0.05 SENSORS=2,1 TAG=_cddnoise run/run_synchronization.py lorenz63
    step ${S}_lorenz63_s3_2_1_a12_cddnoise.npz -- CDD_STD=0.05 SENSORS=3,2,1 N_ANALYSES=1 TAG=_a12_cddnoise run/run_synchronization.py lorenz63
    step ${S}_lorenz96.npz -- CDD_STD=0.2 run/run_synchronization.py lorenz96
    for s in 2 3; do step ${S}_lorenz96_s$s.npz -- CDD_STD=0.2 SENSORS=$s run/run_synchronization.py lorenz96; done
    step ${S}_lorenz96_s10_5_3_a12.npz -- CDD_STD=0.2 SENSORS=10,5,3 N_ANALYSES=1,2 TAG=_a12 run/run_synchronization.py lorenz96
    step ${S}_lorenz96_s10_cddnoise.npz -- CDD_STD=0.02 SENSORS=10 TAG=_cddnoise run/run_synchronization.py lorenz96
    step ${S}_lorenz96_s5_3_2_cddnoise.npz -- CDD_STD=0.02 SENSORS=5,3,2 TAG=_cddnoise run/run_synchronization.py lorenz96
    step ${S}_lorenz96_s10_5_3_a12_cddnoise.npz -- CDD_STD=0.02 SENSORS=10,5,3 N_ANALYSES=1,2 TAG=_a12_cddnoise run/run_synchronization.py lorenz96
    step ${S}_lorenz96_parallel_g5_h2_n500_nx10_w20.npz -- NX=10 G_SITES=5 HALO=2 N_R=500 SENSORS=10,5 run/run_parallel_localization.py sync   # L96-10p, in the text
    figure plot/plot_synchronization.py
    for s in 2 3 5; do step results/partial/partial_lorenz96_$s.jsonl -- CDD_STD=0.2 run/run_lorenz96_small.py partial $s; done
    step -- run/run_verification.py L63
}

parameter() {   # Sect. 5.2 (fig:bifurcation, fig:param, fig:kld) and Appendix B; the inflation of L96-10p; Appendix A, L96-10
    local R=results/readout_parameter P=results/parameter v
    step $R/single1000/train_readout.npz -- run/run_parametric.py train
    step $R/parallel_readout.npz -- run/run_parametric.py train readout parallel
    step $P/lorenz96_bifurcation_Nx10.npz -- run/run_parametric.py truth
    step $R/bifurcation_single1000_readout_peaks.npz -- run/run_parametric.py bifurcation
    step $R/bifurcation_parallel_readout_peaks.npz -- run/run_parametric.py bifurcation readout parallel
    figure plot/plot_l96_bifurcation.py
    step $P/param_scan_lorenz96p_gammainnovation.npz -- DATA_LYAP=30 run/run_parametric.py scan
    step $P/param_scan_lorenz96p_gammainnovation_fixF.npz -- DATA_LYAP=30 FIX_F=1 run/run_parametric.py scan
    step $P/param_attractor_lorenz96p.npz $P/param_kld_perfect_lorenz96p.npz -- DATA_LYAP=30 run/run_parametric.py attractor
    step $P/param_attractor_lorenz96p_fixF.npz -- DATA_LYAP=30 FIX_F=1 run/run_parametric.py attractor
    step $P/param_scan_lorenz96p_gammainnovation_cdd002.npz -- DATA_LYAP=30 CDD_STD=0.02 TAG=_cdd002 run/run_parametric.py scan
    figure plot/plot_parameter_scan.py
    step $R/single1000/one_record.npz -- DATA_LYAP=30 run/run_parametric.py one_record
    for v in paper readout_t12 paper_t12; do step $R/bifurcation_single1000_${v}_peaks.npz -- run/run_parametric.py bifurcation $v; done
    step $R/single1000/scan_readout_decoupled.npz -- DATA_LYAP=30 run/run_parametric.py appscan readout
    step $R/single1000/scan_paper_matched.npz -- DATA_LYAP=30 run/run_parametric.py appscan paper
    step results/uncertainty/uq_parallel10.npz -- run/run_parallel_uq10.py
    step -- run/run_verification.py L96-10
}

localization() {   # Sect. 5.3 (fig:localization, fig:localized_covariance) and 5.4 (fig:parallel_forecast, fig:uncertainty); Appendix A
    local L=results/localization/localization_parallel_lorenz96 W='TEST_WINDOWS=0-19,30-47' c s q
    step -- run/run_parallel_localization.py train
    step -- G_SITES=40 HALO=0 N_R=6000 TRAIN_LYAP=800 run/run_parallel_localization.py train
    step ${L}_w20_innov.npz -- INFLATIONS=innovation $W TAG=_innov run/run_parallel_localization.py run
    step ${L}_w20_innov_more.npz -- INFLATIONS=innovation $W TAG=_innov_more MEMBERS=80,160 RADII=global,2 run/run_parallel_localization.py run
    step ${L}_w20_innov_more5.npz -- INFLATIONS=innovation $W TAG=_innov_more5 MEMBERS=5 run/run_parallel_localization.py run
    step ${L}_g40_h0_n6000_innov.npz -- INFLATIONS=innovation $W G_SITES=40 HALO=0 N_R=6000 TRAIN_LYAP=800 TAG=_innov MEMBERS=5,10,20,40,80,160 run/run_parallel_localization.py run
    step ${L}_w20_innov_cdd1x.npz -- INFLATIONS=innovation $W TAG=_innov_cdd1x CDD_STD=0.02 RADII=2 MEMBERS=5,10,20,40,80,160 run/run_parallel_localization.py run
    figure plot/plot_localization.py
    if [ -n "${FULL:-}" ]; then   # the m = 1000 reference, row (e): about 41 GB, so run it with the machine to itself
        step -- run/run_parallel_localization.py covariance   # runs only the members missing from the record
    else
        step results/localization/localized_covariance.npz results/localization/localized_covariance_positions.npz -- COV_MEMBERS=10 run/run_parallel_localization.py covariance
    fi
    figure plot/plot_localized_covariance.py
    step results/localization/parallel_forecast_windows_cdd0.2.npz results/localization/parallel_forecast_shown.pkl -- run/run_parallel_localization.py forecast
    figure plot/plot_parallel_forecast.py
    if [ -f "$PAPER/parallel_forecast.pdf" ] && [ -z "${DRY_RUN:-}" ] && command -v pdfcrop > /dev/null; then   # as the paper
        pdfcrop --margins 2 "$PAPER/parallel_forecast.pdf" "$PAPER/parallel_forecast.pdf" > /dev/null
    fi
    for c in l63:0.05,0.1,0.25,0.5:1 10s:0.02,0.04,0.1,0.2:5 10p:0.02,0.04,0.1,0.2:5; do   # case:sigma_obs:partial Q
        IFS=: read -r case levels q <<< "$c"
        for s in ${levels//,/ }; do
            step results/uncertainty/uq_levels_${case}_cdd$s.npz -- CDD_STD=$s run/run_uq_levels.py $case
            step results/uncertainty/uq_levels_${case}_cdd${s}_q$q.npz -- CDD_STD=$s Q=$q run/run_uq_levels.py $case
        done
    done
    for s in 0.02 0.04 0.1; do step results/localization/parallel_forecast_windows_cdd$s.npz -- CDD_STD=$s run/run_parallel_localization.py forecast; done
    figure plot/plot_uncertainty.py
    step -- run/run_verification.py L96-40
    figure plot/plot_verification.py             # fig:verification
}


for section in "${@:-synchronization parameter localization}"; do
    for s in $section; do "$s"; done
done
