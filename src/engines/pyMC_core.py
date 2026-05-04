# engines/pymc_core.py
import os
import numpy as np
import pandas as pd

import batman
import multiprocessing as mp

import pymc as pm
import pytensor.tensor as pt
import arviz as az

from pytensor.graph import Op, Apply
from pytensor import config as pt_config

import src.utils.config as con  # keep con.G only
from src.utils.gpu_config import gpu_config_init as gpu_config_func
class BatmanOp(Op):
    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def make_node(self, *inputs):
        # Convert all inputs to tensors if they aren't already
        converted_inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        return Apply(self, converted_inputs, [o() for o in self.otypes])

    def perform(self, node, inputs, outputs):
        time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad = inputs
        params = batman.TransitParams()

        params.t0 = float(t0)
        params.per = float(per)
        params.rp = float(rp_rs)
        params.a = float(a_rs)
        params.inc = float(inc)
        params.u = [float(u1), float(u2)]
        params.ecc = float(ecc)
        params.w = 90.0
        params.u = [float(u1), float(u2)]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, time, supersample_factor=4, exp_time=cad/24./60.)
        outputs[0][0] = m.light_curve(params)

    def grad(self, inputs, g_outputs):
        # For now, return zeros (no gradient)
        return [pt.zeros(inp.shape, dtype=pt_config.floatX) for inp in inputs]
    
gpu_config_func()  # Enable JAX Metal GPU if available


def extract_summary_dataframe(trace, hdi_prob=0.68):
    summary = az.summary(trace, hdi_prob=hdi_prob)

    median_dataset = trace.posterior.median(dim=["chain", "draw"])
    medians = {var: float(median_dataset[var]) for var in median_dataset.data_vars}
    summary["median"] = medians

    selected_columns = ["mean", "median", "sd", "hdi_16%", "hdi_84%", "r_hat"]
    return summary[selected_columns]


def transit_mask_tensors(t, period, duration, T0, cad_minutes=None):
    phase = pt.abs(((t - T0 + 0.5 * period) % period) - (0.5 * period))
    buffer = 0.0
    if cad_minutes is not None:
        buffer = cad_minutes / (24.0 * 60.0)
    return phase < (0.5 * duration + buffer)


def sample_until_converged(model, max_attempts=3, rhat_threshold=1.1, chains=4, cores=None, mp_context="spawn"):
    """Get all free random variables in the model.
    
    Returns:
        trace, attempt: Return values on convergence
    Raises:
        RuntimeError: If model does not converge after max_attempts
    """
    cores = min(chains, os.cpu_count() or 1) if cores is None else cores

    free_vars = model.free_RVs
    if not free_vars:
        raise ValueError("No free random variables found for sampling.")

    step = pm.DEMetropolisZ(vars=free_vars)

    for attempt in range(1, max_attempts + 1):
        print(f"Sampling attempt {attempt}...")
        run = 2 + attempt
        with model:
            trace = pm.sample(
                step=step,
                draws=5000*(run),
                tune=2000*(run),
                chains=chains,
                cores=cores,
                mp_ctx=mp.get_context(mp_context),
                random_seed=list(range(chains)),
                nuts_sampler="pymc",
            )
            # Check convergence
            summary = az.summary(trace)
            if (summary["r_hat"] < rhat_threshold).all():
                print(f"Converged on attempt {attempt}")
                return trace, attempt
            print("checking nanas summary", az.summary(trace))
            print(f"Attempt {attempt} failed to converge.")
        raise RuntimeError("Model did not converge after multiple attempts.")


def prepare_fit_data(time, flux, unc, candidate):
    """Filter NaN values and calculate cadence. Returns: time, flux, unc, cad."""
    mask = np.logical_or(np.logical_or(np.isnan(flux), np.isnan(time)), np.isnan(unc))
    time = time[~mask]
    unc = unc[~mask]
    flux = flux[~mask]
    cad = np.nanpercentile(np.clip(np.diff(np.unique(time))*60.*24., 200/60, 30), 95)
    return time, flux, unc, float(cad)

def set_up_variables_for_pymc_fit(time, flux, unc, t0, other_pars, type_fn):
    """
    Returns:
        use_time, use_flux, use_unc, per, u1, u2, depth, cad

    cad is a robust effective cadence in MINUTES.
    """
    mask = np.logical_or(np.logical_or(np.isnan(flux),  np.isnan(time)),  np.isnan(unc))
    time = time[~mask]
    unc  = unc[~mask]
    flux = flux[~mask]

    cad = np.nanpercentile(np.clip(np.diff(np.unique(time))*60.*24., 200/60, 30), 95) #minutes

    if type_fn == 'Single':
        dur, ab, depth = other_pars
        k = np.sqrt(depth)
        per1 = np.max(time)-np.min(time)
        per2 = ((3*np.pi/con.G/con.rho_star)** 0.5 ) * (dur/np.pi/(1+k)) ** 1.5
        per = np.max([per1, per2])
        print('time difference', np.max(time)-np.min(time), 'checking duration units (must be < 0.5 for days, <12 for hours)', dur)
        if per<10:
            per = 27.8
        
        indxes = np.where(np.abs(time-t0)<(1.+dur))
        
        use_time = np.array(time)[indxes]
        use_flux = np.array(flux)[indxes]
        use_unc  = np.array(unc)[indxes]
    elif type_fn == 'Periodic':
        per, ab, depth = other_pars
        use_time = np.array(time)
        use_flux = np.array(flux)
        use_unc  = np.array(unc)
        
        print(per, type(per))
        
        a_smaj_guess = float((con.G * con.rho_star * (per ** 2) / 3 / np.pi) ** (1/3))
        dur =  min([0.5/3, (float(per) / float(a_smaj_guess * np.pi))])  # days

    u1, u2 = ab

    return use_time, use_flux, use_unc, float(per), u1, u2, float(depth), 1.5*float(dur), cad


def median_pytensor(x):
    sorted_x = pt.sort(x)
    n = x.shape[0]
    mid = n // 2
    return pt.switch(
        pt.eq(n % 2, 0),
        (sorted_x[mid - 1] + sorted_x[mid]) / 2.0,
        sorted_x[mid]
    )



def make_windows_from_time_stamps(t, gap_threshold=0.5):
    """
    Convert sorted time stamps (days) into contiguous [start, end] windows.
    gap_threshold is the minimum gap that splits windows (days).
    Pick a value larger than cadence and smaller than any real gap.
    """
    t = np.asarray(t)
    t = t[np.isfinite(t)]
    t = np.sort(t)
    if t.size == 0:
        return np.empty((0, 2))
    gaps = np.diff(t)
    breaks = np.where(gaps > gap_threshold)[0]
    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def make_node(self, *inputs):
        # Convert all inputs to tensors if they aren't already
        converted_inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        return Apply(self, converted_inputs, [o() for o in self.otypes])

    def perform(self, node, inputs, outputs):
        time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad = inputs
        params = batman.TransitParams()

        params.t0 = float(t0)
        params.per = float(per)
        params.rp = float(rp_rs)
        params.a = float(a_rs)
        params.inc = float(inc)
        params.u = [float(u1), float(u2)]
        params.ecc = float(ecc)
        params.w = 90.0
        params.u = [float(u1), float(u2)]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, time, supersample_factor=4, exp_time=cad/24./60.)
        outputs[0][0] = m.light_curve(params)

    def grad(self, inputs, g_outputs):
        # For now, return zeros (no gradient)
        return [pt.zeros(inp.shape, dtype=pt_config.floatX) for inp in inputs]
    

        summary = az.summary(trace)
        if (summary["r_hat"] < rhat_threshold).all():
            print(f"Converged on attempt {attempt}")
            return trace, attempt
        print("checking nanas summary", az.summary(trace))
        print(f"Attempt {attempt} failed to converge.")

    raise RuntimeError("Model did not converge after multiple attempts.")



def make_windows_from_time_stamps(t, gap_threshold=0.5):
    """
    Convert sorted time stamps (days) into contiguous [start, end] windows.
    gap_threshold is the minimum gap that splits windows (days).
    Pick a value larger than cadence and smaller than any real gap.
    """
    t = np.asarray(t)
    t = t[np.isfinite(t)]
    t = np.sort(t)
    if t.size == 0:
        return np.empty((0, 2))
    gaps = np.diff(t)
    breaks = np.where(gaps > gap_threshold)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends    = np.concatenate((breaks, [t.size - 1]))
    return np.column_stack((t[starts], t[ends]))


def pymc_fit_candidate(target, candidate, time, flux, unc, verbose=False, keep_ld_fixed=True):
    # --- star facts from Target ---
    if target.rho_star is None:
        raise ValueError("target.rho_star is None. Ensure catalog Mass/Rad exist and rho_star was computed.")
    rho_star = float(target.rho_star)

    u1, u2 = target.ld_u1_u2

    # --- hypothesis from Candidate ---
    type_fn = candidate.ptype
    T0 = float(candidate.t0_days)
    Depth = float(candidate.depth)

    # periodic requires a period
    Per_in = getattr(candidate, "period_days", None)
    if type_fn == "Periodic":
        if Per_in is None:
            raise ValueError("Periodic candidate missing period_days.")
        Per_in = float(Per_in)


    # Always do data prep first so cad exists
    time, flux, unc, cad = prepare_fit_data(time, flux, unc, candidate)

    # Then validate
    if candidate.depth is None or candidate.duration_days is None:
        raise ValueError("Candidate missing depth or duration_days.")

    # pTdur is your "window scale". Keep your old convention unless you add a dedicated field later.
    pTdur = 1.5 * float(candidate.duration_days)

    batman_op = BatmanOp()

    # count observed transits. Override from candidate if available.
    nobs_est = None
    if type_fn == "Periodic":
        nobs_est = getattr(candidate, "n_transits_obs", None)
        print('nobs_est from candidate:', nobs_est)
        if nobs_est is None:
            windows = make_windows_from_time_stamps(np.array(time), gap_threshold=0.5)
            tmp = 0
            for s, e in windows:
                k_low   = np.ceil((s - T0) / Per_in)
                k_high = np.floor((e - T0) / Per_in)
                tmp += int(max(0, k_high - k_low + 1))
            nobs_est = tmp
        nobs_est = int(nobs_est)

    ecc = 0.0

    fold_this = False  # default for Single; may be updated in model setup

    with pm.Model() as model:
        t0 = pm.Uniform("t0", lower=T0 - pTdur, upper=T0 + pTdur)

        if type_fn == "Single":
            # initial a/R* guess uses a period guess
            k = np.sqrt(Depth)
            per1 = float(np.max(time) - np.min(time))
            per2 = float(((3*np.pi / con.G / rho_star)**0.5) * (pTdur/np.pi/(1+k))**1.5)
            Per_guess = max(per1, per2)
            if Per_guess < 10:
                Per_guess = 27.8

            a_rs_init = float(((con.G * rho_star * (Per_guess ** 2)) / (3.0 * np.pi)) ** (1.0 / 3.0))
            a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_init, sigma=5.0, lower=1.0, initval=a_rs_init)

            per = pm.Deterministic("Per", pt.sqrt((3.0 * np.pi) / (con.G * rho_star)) * a_rs ** 1.5)

        else:   # Periodic
            Per = Per_in
            if nobs_est >= 3:
                per = pm.Uniform("Per", lower=max(0.25, Per * 0.99), upper=Per * 1.01)
                a_rs = pm.Uniform("a_rs", lower=1.0, upper=300.0)
                fold_this = True
            else:
                per = pm.TruncatedNormal("Per", mu=Per, sigma=max(0.1, 0.05 * Per),
                                         lower=max(0.25, Per * 0.80), upper=Per * 1.20)
                a_rs_mu = pm.Deterministic("a_rs_mu", (con.G * rho_star * (per ** 2) / (3*np.pi)) ** (1/3))
                a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_mu, sigma=3.0, lower=1.0)

        # geometry and LD from Target
        eps = 1e-12

        rp_rs = pm.TruncatedNormal("rp_rs", mu=pt.sqrt(Depth),
                                   sigma=pt.maximum(0.02, 0.5 * pt.sqrt(Depth)),
                                   lower=0, upper=1)
        b = pm.TruncatedNormal("b", mu=0, sigma=0.01, lower=0, upper=1)

        depth = pm.Deterministic("depth", rp_rs**2)

        cosi = pm.Deterministic("cosi", pt.clip(b / a_rs, -1.0 + eps, 1.0 - eps))
        inc   = pm.Deterministic("inclination", pt.arccos(cosi) * 180.0 / np.pi)

        root = pt.sqrt(pt.clip(1.0 - b**2, eps, 1.0))
        T_dur0 = per / ((a_rs + eps) * np.pi)
        tau = pm.Deterministic("tau", rp_rs * T_dur0 / root)
        dur = pm.Deterministic("dur", root * T_dur0 + tau)
        win = pm.Deterministic("win", dur * 2.0)

        # masks
        if type_fn == "Periodic":
            intran_mask = transit_mask_tensors(time, per, dur, t0, cad)
        else:
            intran_mask = pt.abs(time - t0) < (dur / 2.0)

        outran_mask = pt.invert(intran_mask)

        out_flux = flux * outran_mask
        count = pt.maximum(pt.sum(outran_mask), 1)
        mean_out = pt.sum(out_flux) / count
        std_out = pt.sqrt(pt.sum(outran_mask * (flux - mean_out)**2) / count)

        N_tran = pt.sum(intran_mask)
        uq = pt.ones_like(flux) * std_out
        sigs = pt.switch(N_tran > 0, pt.mean(pt.where(intran_mask, uq, 0)), 1e6)

        SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0)
        SNR_clipped = pt.clip(SNR_val, 0, 1e4)
        SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)
        if not fold_this:
            pm.Deterministic("SNR", SNR_final)

        norm = pm.Deterministic("norm", median_pytensor(out_flux))

        if fold_this:
            print('phase folded')
            folded_phase = ((time - T0 + 0.5 * Per_in) % Per_in) - (0.5 * Per_in)
            sort_indx = np.argsort(folded_phase)
            phase = folded_phase[sort_indx]
            use_index = np.abs(phase) < min([0.5, 3*pTdur])

            dt_minutes_min = np.nanpercentile(np.diff(np.unique(np.sort(time))), 5) * 24.0 * 60.0
            p_cad = float(np.clip(dt_minutes_min, 0.2, 60.0))

            p_flux_model = batman_op(phase[use_index] + T0, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, p_cad)
            pm.Normal("obs", mu=p_flux_model * norm, sigma=unc[sort_indx][use_index], observed=flux[sort_indx][use_index])
        else:
            flux_model = batman_op(time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad)
            pm.Normal("obs", mu=flux_model * norm, sigma=unc, observed=flux)

    with model:
        try:
            trace, conv_attempt = sample_until_converged(model)
            summary = extract_summary_dataframe(trace)
        except RuntimeError:
            return pd.DataFrame(columns=["mean","median","sd","hdi_16%","hdi_84%","r_hat"]), False, None

    # Keep plots off in core. The caller decides.
    return summary, True, conv_attempt
