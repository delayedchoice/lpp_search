from engines.pymc_core import pymc_fit_candidate

def fit_and_attach(target, candidate, time, flux, unc, verbose=False):
    summary_df, ok, _ = pymc_fit_candidate(target, candidate, time, flux, unc, verbose=verbose)
    if ok and summary_df is not None:
        candidate.pymc_summary = summary_df.to_dict()
        candidate.mark_fitted()
        return True
    candidate.fit_is_current = False
    return False