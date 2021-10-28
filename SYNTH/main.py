# code for reproducing plots related to synthetic experiments.
# just execute this file, it will save plots in the execution folder.


from time import time, sleep
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from SYNTH import distributions, imle, ste, utils, sfe

FIGSIZE = (3.2, 2.5)

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def optim_loop(min_objective, true_objective, lr, momentum_factor,
               n_steps, theta0, _debug_print_loss=False):

    theta_t = t.from_numpy(theta0).float().requires_grad_(True)
    # let's try to optimize this expectation w.r.t. theta
    _opt = t.optim.SGD([theta_t], lr, momentum=momentum_factor)
    _hist, _hist_expectation = [], []
    for _t in range(n_steps):
        _opt.zero_grad()
        _obj = min_objective(theta_t)
        if _debug_print_loss: print(_obj)
        _hist.append(_obj.detach().numpy())
        _hist_expectation.append(true_objective(theta_t).detach().numpy())
        _obj.backward()
        _opt.step()
    return _hist, _hist_expectation


def experiment(min_obj, ture_objective, lr, theta0, momentum=0.9,
               steps=50, n_rp=50, do_plot=True, postprocess=None):
    # redefine objective with given strategy
    hist = []
    for _ in range(n_rp):
        print('-', end='')
        stoc_obj, true_obj = optim_loop(min_obj, ture_objective, lr, momentum, steps,
                                        theta0)
        if postprocess:
            true_obj = postprocess(true_obj)
        hist.append(true_obj)


    if do_plot:
        mean = np.mean(hist, axis=0)
        # plt.plot(full_optim_hist)
        plt.plot(mean)
        plt.show()

    print()
    return hist


def plot_mean_std(histories, names, xs=None):
    means = [np.mean(np.array(his), axis=0) for his in histories]
    std_devs = [np.std(np.array(his), axis=0) for his in histories]

    for h, st, nm in zip(means, std_devs, names):
        x_axis = xs if xs else list(range(len(h)))
        line = plt.plot(xs, h, label=nm)
        plt.fill_between(x_axis, h - st, h + st, alpha=0.5,
                         color=line[0].get_color())


def toy_exp(n, k, n_rep=50, an=''):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    exp = lambda strategy, lr, n_rp=n_rep, steps=50: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp
    )

    def do_plots(histories, names, savename=None, figsize=FIGSIZE):
        # computing also standard devs
        plt.figure(figsize=figsize)
        means = [np.mean(np.array(his) - min_value_of_exp, axis=0) for his in histories]
        std_devs = [np.std(np.array(his), axis=0) for his in histories]

        for h, st, nm in zip(means, std_devs, names):
            x_axis = list(range(len(h)))
            line = plt.plot(h, label=nm)
            plt.fill_between(x_axis, h - st, h + st, alpha=0.5,
                             color=line[0].get_color())

        plt.legend(loc=0)
        plt.ylim((0., 3.))
        plt.xlim((0, 49))
        plt.xlabel('Optimization steps')
        plt.ylabel('Optimality gap')
        if savename:
            plt.savefig(savename, bbox_inches='tight')
        plt.show()

    pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
    pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

    # hyperparameter values obtained by grid search (sensitivity_imle)
    imle_pam_lcs = exp(imle.imle_pid(2., pam_sog), 0.75)

    # hyperparameter values obtained by grid search (sensitivity_ste)
    ste_pam_lcs = exp(ste.ste(pam_gum), 0.019)

    do_plots([ste_pam_lcs, imle_pam_lcs],
             ['STE PaM', 'I-MLE PaM'],
             figsize=(4, 3))

    # hyperparameter values obtained by grid search (sensitivity_sfe)
    sfe_full = sfe.sfe(topk.sample_f(np.random.RandomState(0)),
                       objective, topk.grad_log_p(topk.marginals))
    sfe_full_lcs = exp(sfe_full, .0046, n_rp=n_rep//5, steps=500)
    ary_sfe = np.array(sfe_full_lcs)

    # final plot!
    do_plots([ary_sfe[:, ::10], ste_pam_lcs, imle_pam_lcs],
             ['SFE (steps x 10)', r'STE', r'I-MLE'])


def exp_computing_time(n_rep=10):
    ns = [8, 10, 12, 14, 16, 18, 20, 22, 24]
    rng = np.random.RandomState(0)
    t_full_obj, t_pam, t_samp = np.zeros((n_rep, len(ns))), \
                                np.zeros((n_rep, len(ns))), np.zeros((n_rep, len(ns)))
    for i, n in enumerate(ns):
        k = n//2
        theta = t.from_numpy(rng.randn(n)).float()
        topk = distributions.TopK(n, k)
        b_t = t.abs(t.from_numpy(rng.randn(n)).float())

        def objective(z): return ((z - b_t)**2).sum()
        full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

        print('full objective')
        sleep(2)
        for r in range(n_rep):
            if n <=18:
                t0 = time()
                full_obj(theta)
                t_full_obj[r, i] = time() - t0
            else:
                t_full_obj[r, i] = 2.

        pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k))
        # pam_sog = topk.map
        strategy = imle.imle_pid(1., pam_sog)
        pam_objective = lambda _th: ((strategy(_th) - b_t)**2).sum()

        print('pam:')
        sleep(2)
        for r in range(n_rep):
            t0 = time()
            pam_objective(theta)
            t_pam[r, i] = time() - t0

        sample_obj = lambda _th: ((topk.sample(_th) - b_t)**2).sum()

        # topk._states = None
        print('sample')
        sleep(2)
        for r in range(n_rep):
            t0 = time()
            sample_obj(theta)
            t_samp[r, i] = time() - t0

    plt.figure(figsize=FIGSIZE)
    # plt.yscale('log')
    plot_mean_std([t_full_obj, t_samp, t_pam], ['Expect.', 'Sample', 'P&M'], xs=ns)
    plt.legend(loc=0)
    plt.ylabel('Seconds')
    plt.xlabel('Size (m)')
    plt.ylim((-0.02, .6))
    plt.xlim((ns[0], ns[-1]))
    plt.savefig('times.pdf', bbox_inches='tight')
    plt.show()


def sensibility_imle(n, k, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=50: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 5, 6

    search_grid_lr = np.linspace(0.5, 1., num=n_lr)
    search_grid_lambda = np.linspace(0.5, 3., num=n_lbd)

    res_sog_mean, res_sog_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    res_gum_mean, res_gum_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            print(i, j)
            pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
            pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

            imle_sog_lcs = exp(imle.imle_pid(lmd, pam_sog), lr)
            imle_gum_lcs = exp(imle.imle_pid(lmd, pam_gum), lr)

            res_sog_mean[i, j] = np.mean(np.array(imle_sog_lcs) - min_value_of_exp, axis=0)[-1]
            res_sog_std[i, j] = np.std(np.array(imle_sog_lcs), axis=0)[-1]

            res_gum_mean[i, j] = np.mean(np.array(imle_gum_lcs) - min_value_of_exp, axis=0)[-1]
            res_gum_std[i, j] = np.std(np.array(imle_gum_lcs), axis=0)[-1]

    def do_plot(what, name, xlabel='Lambda', ylabel='Learning rate', nm=False):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        if not nm:
            pos = ax.imshow(what, cmap='hot_r', interpolation='nearest')
        else:
            elev_min = np.min(what)
            elev_max = np.max(what)
            mid_val = 0.
            pos = ax.imshow(what, cmap='seismic',  interpolation='nearest',
                            clim=(elev_min, elev_max),
                            norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
        ax.set_yticks(list(range(len(search_grid_lr))))
        ax.set_yticklabels(search_grid_lr)

        ax.set_xticks(list(range(len(search_grid_lambda))))
        ax.set_xticklabels(search_grid_lambda)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        fig.colorbar(pos, ax=ax)
        if name:
            plt.savefig(f'cm_{name}.pdf', bbox_inches='tight')
        plt.show()

    do_plot(res_sog_mean, 'I-MLE PaM SoG')
    do_plot(res_gum_mean, 'I-MLE PaM Gumbel')
    do_plot(res_sog_mean - res_gum_mean, 'I-MLE SoG - Gum. (means)', nm=True)

    do_plot(res_sog_std, 'I-MLE PaM SoG - std')
    do_plot(res_gum_std, 'I-MLE PaM Gumbel - std')
    do_plot(res_sog_std - res_gum_std, 'I-MLE SoG - Gum. (std)', nm=True)


def sensibility_ste(n, k, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=50: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 10, 1

    search_grid_lr = np.exp(np.linspace(np.log(0.001), np.log(.2), num=n_lr))
    search_grid_lambda = np.linspace(0.5, 3., num=n_lbd)

    res_ste_mean, res_ste_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    res_ste_g_mean, res_ste_g_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))
    # res_gum_mean, res_gum_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            print(i, j)
            pam_sog = topk.perturb_and_map(utils.sum_of_gamma_noise(k, rng=np.random.RandomState(0)))
            pam_gum = topk.perturb_and_map(utils.gumbel_noise(rng=np.random.RandomState(0)))

            ste_pam_lcs = exp(ste.ste(pam_sog), lr)
            ste_pam_g_lcs = exp(ste.ste(pam_gum), lr)

            res_ste_mean[i, j] = np.mean(np.array(ste_pam_lcs) - min_value_of_exp, axis=0)[-1]
            # res_sog_std[i, j] = np.std(np.array(imle_sog_lcs), axis=0)[-1]

            res_ste_g_mean[i, j] = np.mean(np.array(ste_pam_g_lcs) - min_value_of_exp, axis=0)[-1]
            # res_gum_std[i, j] = np.std(np.array(imle_gum_lcs), axis=0)[-1]

    def do_plot(what, name, xlabel='', ylabel='Learning rate', nm=False):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        if not nm:
            pos = ax.imshow(what, cmap='hot_r', interpolation='nearest')
        else:
            elev_min = np.min(what)
            elev_max = np.max(what)
            mid_val = 0.
            pos = ax.imshow(what, cmap='seismic',  interpolation='nearest',
                            clim=(elev_min, elev_max),
                            norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
        ax.set_yticks(list(range(len(search_grid_lr))))
        ax.set_yticklabels(['{:.4f}'.format(lr) for lr in search_grid_lr])

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        fig.colorbar(pos, ax=ax)
        if name:
            plt.savefig(f'cm_{name}.pdf', bbox_inches='tight')
        plt.show()

    do_plot(res_ste_mean, 'STE PaM SoG')
    do_plot(res_ste_g_mean, 'STE PaM Gumbel')
    do_plot(res_ste_mean - res_ste_g_mean, 'STE SoG - Gum. (means)', nm=True)


def sensibility_sfe(n, k, n_rep=20):
    rng = np.random.RandomState(0)
    theta = rng.randn(n)
    topk = distributions.TopK(n, k)

    b_t = t.abs(t.from_numpy(rng.randn(n)).float())
    print(b_t)

    sorted_bt = np.sort(b_t.detach().numpy())
    min_value_of_exp = np.sum((sorted_bt[:k])**2) + np.sum((sorted_bt[k:] - 1)**2)
    print(min_value_of_exp)

    def objective(z):
        return ((z - b_t)**2).sum()

    full_obj = lambda _th: utils.expect_obj(topk, _th, objective)

    def pp(_his):
        if _his[-1] - min_value_of_exp < 0.:  # then it's all lost
            _his[-1] = 5.
            print('pp')
        return _his

    exp = lambda strategy, lr, n_rp=n_rep, steps=50: experiment(
        lambda _th: ((strategy(_th) - b_t)**2).sum(),
        full_obj,
        lr, theta, steps=steps, n_rp=n_rp, do_plot=False,
        postprocess=pp
    )

    n_lr, n_lbd = 10, 1

    search_grid_lr = np.exp(np.linspace(np.log(0.0001), np.log(.1), num=n_lr))
    search_grid_lambda = np.linspace(0.5, 3., num=n_lbd)

    res_sfe_mean, res_sfe_std = np.zeros((n_lr, n_lbd)), np.zeros((n_lr, n_lbd))

    for i, lr in enumerate(search_grid_lr):
        for j, lmd in enumerate(search_grid_lambda):
            print(i, j)
            sfe_full = sfe.sfe(topk.sample_f(np.random.RandomState(0)),
                               objective, topk.grad_log_p(topk.marginals))
            sfe_full_lcs = exp(sfe_full, lr, n_rp=n_rep, steps=500)

            res_sfe_mean[i, j] = np.mean(np.array(sfe_full_lcs) - min_value_of_exp, axis=0)[-1]
            res_sfe_std[i, j] = np.std(np.array(sfe_full_lcs), axis=0)[-1]

    def do_plot(what, name, xlabel='', ylabel='Learning rate', nm=False):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        if not nm:
            pos = ax.imshow(what, cmap='hot_r', interpolation='nearest')
        else:
            elev_min = np.min(what)
            elev_max = np.max(what)
            mid_val = 0.
            pos = ax.imshow(what, cmap='seismic',  interpolation='nearest',
                            clim=(elev_min, elev_max),
                            norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
        ax.set_yticks(list(range(len(search_grid_lr))))
        ax.set_yticklabels(['{:.4f}'.format(lr) for lr in search_grid_lr])

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        fig.colorbar(pos, ax=ax)
        if name:
            plt.savefig(f'cm_{name}.pdf', bbox_inches='tight')
        plt.show()

    do_plot(res_sfe_mean, 'SFE SM mean')
    do_plot(res_sfe_std, 'SFE SM std')


if __name__ == '__main__':
    exp_computing_time(10)
    sensibility_imle(10, 5, n_rep=100)
    sensibility_ste(10, 5, n_rep=100)
    sensibility_sfe(10, 5, n_rep=20)
    toy_exp(10, 5, n_rep=100, an='_final')
