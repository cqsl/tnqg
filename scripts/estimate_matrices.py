import numpy as np
from scipy.special import logsumexp


def estimate_matrices(
    distribution, psi0, logphis_model, variables, n_models, H, obs=None, window=128
):
    X = distribution.samples
    X = X.reshape(-1, X.shape[-1])

    n_reps = X.shape[0] // window
    if obs is None:
        obs_list = []
    elif isinstance(obs, (list, tuple)):
        obs_list = list(obs)

    for i in range(n_reps):
        x = X[i * window : (i + 1) * window]
        obs_xp = []
        obs_mels = []
        obs_xps = []
        for obs_i in obs_list:
            xp, mels = obs_i.get_conn_padded(x)
            obs_xp.append(xp)
            obs_mels.append(mels)
            obs_xps.append(np.tile(xp[None, ...], (n_models,) + (1,) * xp.ndim))
        xp_H, mels_H = H.get_conn_padded(x)
        xs = np.tile(x[None, ...], (n_models,) + (1,) * x.ndim)
        xps_H = np.tile(xp_H[None, ...], (n_models,) + (1,) * xp_H.ndim)

        if i == 0:
            logphi_i_x = logphis_model.apply(
                variables,
                xs,
            )  # (M, Ns)
            logpsi0_i_x = psi0._apply_fun(psi0.variables, x)  # (Ns)

            if obs_list:
                logphi_i_xp_list = []
                logpsi0_i_xp_list = []
                for xp, mels, xps in zip(obs_xp, obs_mels, obs_xps):
                    logphi_i_xp = logphis_model.apply(
                        variables,
                        xps,
                    )  # (M, Ns, Nconn)

                    logphi_i_xp = logsumexp(logphi_i_xp, b=mels, axis=-1)  # (M, Ns)

                    logpsi0_i_xp = psi0._apply_fun(psi0.variables, xp)  # (Ns, Nconn)
                    logpsi0_i_xp = logsumexp(logpsi0_i_xp, b=mels, axis=-1)  # (Ns)
                    logphi_i_xp_list.append(logphi_i_xp)
                    logpsi0_i_xp_list.append(logpsi0_i_xp)

            logphi_i_xp_H = logphis_model.apply(
                variables,
                xps_H,
            )  # (M, Ns, Nconn)
            logphi_i_xp_H = logsumexp(logphi_i_xp_H, b=mels_H, axis=-1)  # (M, Ns)

            logpsi0_i_xp_H = psi0._apply_fun(psi0.variables, xp_H)  # (Ns, Nconn)
            logpsi0_i_xp_H = logsumexp(logpsi0_i_xp_H, b=mels_H, axis=-1)  # (Ns)

            logP = 2 * distribution._apply_fun(distribution.variables, x).real  # (Ns)

        else:
            logphi_i_x_ = logphis_model.apply(
                variables,
                xs,
            )  # (M, Ns)
            logpsi0_i_x_ = psi0._apply_fun(psi0.variables, x)  # (Ns)

            if obs_list:
                logphi_i_xp_list_ = []
                logpsi0_i_xp_list_ = []
                for xp, mels, xps in zip(obs_xp, obs_mels, obs_xps):
                    logphi_i_xp_ = logphis_model.apply(
                        variables,
                        xps,
                    )  # (M, Ns, Nconn)
                    logphi_i_xp_ = logsumexp(logphi_i_xp_, b=mels, axis=-1)  # (M, Ns)
                    logpsi0_i_xp_ = psi0._apply_fun(psi0.variables, xp)  # (Ns, Nconn)
                    logpsi0_i_xp_ = logsumexp(logpsi0_i_xp_, b=mels, axis=-1)  # (Ns)
                    logphi_i_xp_list_.append(logphi_i_xp_)
                    logpsi0_i_xp_list_.append(logpsi0_i_xp_)

            logphi_i_xp_H_ = logphis_model.apply(
                variables,
                xps_H,
            )  # (M, Ns, Nconn)
            logphi_i_xp_H_ = logsumexp(logphi_i_xp_H_, b=mels_H, axis=-1)  # (M, Ns)
            logpsi0_i_xp_H_ = psi0._apply_fun(psi0.variables, xp_H)  # (Ns, Nconn)
            logpsi0_i_xp_H_ = logsumexp(logpsi0_i_xp_H_, b=mels_H, axis=-1)  # (Ns)

            logP_ = 2 * distribution._apply_fun(distribution.variables, x).real  # (Ns)

            logphi_i_x = np.concatenate((logphi_i_x, logphi_i_x_), axis=1)  # (M, Ns)
            logpsi0_i_x = np.concatenate((logpsi0_i_x, logpsi0_i_x_), axis=0)  # (Ns)

            if obs_list:
                for j in range(len(obs_list)):
                    logphi_i_xp_list[j] = np.concatenate(
                        (logphi_i_xp_list[j], logphi_i_xp_list_[j]), axis=1
                    )  # (M, Ns)
                    logpsi0_i_xp_list[j] = np.concatenate(
                        (logpsi0_i_xp_list[j], logpsi0_i_xp_list_[j]), axis=0
                    )  # (Ns)

            logphi_i_xp_H = np.concatenate(
                (logphi_i_xp_H, logphi_i_xp_H_), axis=1
            )  # (M, Ns)
            logpsi0_i_xp_H = np.concatenate(
                (logpsi0_i_xp_H, logpsi0_i_xp_H_), axis=0
            )  # (Ns)

            logP = np.concatenate((logP, logP_), axis=0)  # (Ns)

    logphi_i_x = np.concatenate(
        (np.expand_dims(logpsi0_i_x, axis=0), logphi_i_x), axis=0
    )  # (M, Ns)
    if obs_list:
        for j in range(len(obs_list)):
            logphi_i_xp_list[j] = np.concatenate(
                (np.expand_dims(logpsi0_i_xp_list[j], axis=0), logphi_i_xp_list[j]),
                axis=0,
            )  # (M, Ns)

    logphi_i_xp_H = np.concatenate(
        (np.expand_dims(logpsi0_i_xp_H, axis=0), logphi_i_xp_H), axis=0
    )  # (M, Ns)

    logP = np.expand_dims(logP, axis=(0, 1))  # (1, 1, Ns)

    logS_ki = np.conj(logphi_i_x)[:, None, :] + logphi_i_x[None, :, :]  # (M, M, Ns)
    logS_ki = logS_ki - logP
    logS_ki = logsumexp(logS_ki, axis=-1) - np.log(
        logS_ki.shape[-1]
    )  # (M, M)  log of the mean
    S_ki = np.exp(logS_ki)

    if obs_list:
        O_ki_list = []
        for logphi_i_xp in logphi_i_xp_list:
            logO1 = (
                np.conj(logphi_i_xp)[:, None, :] + logphi_i_x[None, :, :] - logP
            )  # (M, M, Ns)
            logO2 = (
                np.conj(logphi_i_x)[:, None, :] + logphi_i_xp[None, :, :] - logP
            )  # (M, M, Ns)
            logO_ki = logsumexp(np.stack([logO1, logO2], axis=0), axis=0) - np.log(
                2
            )  # (M,M,Ns)
            logO_ki = logsumexp(logO_ki, axis=-1) - np.log(logO_ki.shape[-1])  # (M,M)
            O_ki = np.exp(logO_ki)  # (M,M,Ns)

            O_ki_list.append(O_ki)  # (M, M)

    logF1 = (
        np.conj(logphi_i_xp_H)[:, None, :] + logphi_i_x[None, :, :] - logP
    )  # (M, M, Ns)
    logF2 = (
        np.conj(logphi_i_x)[:, None, :] + logphi_i_xp_H[None, :, :] - logP
    )  # (M, M, Ns)
    logF_ki = logsumexp(np.stack([logF1, logF2], axis=0), axis=0) - np.log(
        2
    )  # (M,M,Ns)
    logF_ki = logsumexp(logF_ki, axis=-1) - np.log(logF_ki.shape[-1])  # (M,M)
    F_ki = np.exp(logF_ki)  # (M,M,Ns)

    if obs is None:
        return S_ki, F_ki
    else:
        return S_ki, O_ki_list, F_ki
