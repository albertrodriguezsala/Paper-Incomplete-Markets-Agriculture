# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:37:00 2020

@author: from quantecon code add saving option
"""
import time
import warnings
import numpy as np


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.")





def _print_after_skip(skip, it=None, dist=None, etime=None):
    if it is None:
        # print initial header
        msg = "{i:<13}{d:<15}{t:<17}".format(i="Iteration",
                                             d="Distance",
                                             t="Elapsed (seconds)")
        print(msg)
        print("-" * len(msg))

        return

    if it % skip == 0:
        if etime is None:
            print("After {it} iterations dist is {d}".format(it=it, d=dist))

        else:
            # leave 4 spaces between columns if we have %3.3e in d, t
            msg = "{i:<13}{d:<15.3e}{t:<18.3e}"
            print(msg.format(i=it, d=dist, t=etime))

    return


_convergence_msg = 'Converged in {iterate} steps'
_non_convergence_msg = \
    'max_iter attained before convergence in compute_fixed_point'


def _is_approx_fp(T, v, error_tol, *args, **kwargs):
    error = np.max(np.abs(T(v, *args, **kwargs) - v))
    return error <= error_tol



def compute_fixed_point(T, v, error_tol=1e-3, max_iter=50, verbose=2,
                        print_skip=5, method='iteration', save=True, save_name='v_itera5', folder='C:/Users/rodri/Dropbox/JMP/figures/model economy',  *args, **kwargs):
    r"""
    Computes and returns an approximate fixed point of the function `T`.
    The default method `'iteration'` simply iterates the function given
    an initial condition `v` and returns :math:`T^k v` when the
    condition :math:`\lVert T^k v - T^{k-1} v\rVert \leq
    \mathrm{error\_tol}` is satisfied or the number of iterations
    :math:`k` reaches `max_iter`. Provided that `T` is a contraction
    mapping or similar, :math:`T^k v` will be an approximation to the
    fixed point.
    The method `'imitation_game'` uses the "imitation game algorithm"
    developed by McLennan and Tourky [1]_, which internally constructs
    a sequence of two-player games called imitation games and utilizes
    their Nash equilibria, computed by the Lemke-Howson algorithm
    routine. It finds an approximate fixed point of `T`, a point
    :math:`v^*` such that :math:`\lVert T(v) - v\rVert \leq
    \mathrm{error\_tol}`, provided `T` is a function that satisfies the
    assumptions of Brouwer's fixed point theorm, i.e., a continuous
    function that maps a compact and convex set to itself.
    Parameters
    ----------
    T : callable
        A callable object (e.g., function) that acts on v
    v : object
        An object such that T(v) is defined; modified in place if
        `method='iteration' and `v` is an array
    error_tol : scalar(float), optional(default=1e-3)
        Error tolerance
    max_iter : scalar(int), optional(default=50)
        Maximum number of iterations
    verbose : scalar(int), optional(default=2)
        Level of feedback (0 for no output, 1 for warnings only, 2 for
        warning and residual error reports during iteration)
    print_skip : scalar(int), optional(default=5)
        How many iterations to apply between print messages (effective
        only when `verbose=2`)
    method : str, optional(default='iteration')
        str in {'iteration', 'imitation_game'}. Method of computing
        an approximate fixed point
    args, kwargs :
        Other arguments and keyword arguments that are passed directly
        to  the function T each time it is called
    Returns
    -------
    v : object
        The approximate fixed point
    References
    ----------
    .. [1] A. McLennan and R. Tourky, "From Imitation Games to
       Kakutani," 2006.
    """
    if max_iter < 1:
        raise ValueError('max_iter must be a positive integer')

    if verbose not in (0, 1, 2):
        raise ValueError('verbose should be 0, 1 or 2')

    if method not in ['iteration', 'imitation_game']:
        raise ValueError('invalid method')

    

    # method == 'iteration'
    iterate = 0

    if verbose == 2:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    while True:
        new_v = T(v, *args, **kwargs)
        iterate += 1
        error = np.max(np.abs(new_v - v))
        
        if save==True:
            np.save(folder+save_name,new_v)

        try:
            v[:] = new_v
        except TypeError:
            v = new_v

        if error <= error_tol or iterate >= max_iter:
            break

        if verbose == 2:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

    if verbose == 2:
        etime = time.time() - start_time
        print_skip = 1
        _print_after_skip(print_skip, iterate, error, etime)
    if verbose >= 1:
        if error > error_tol:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return v





def fixed_point_policies(T, policies, error_tol=1e-3, max_iter=50, verbose=2,
                        print_skip=5, method='iteration', save=True, save_name='v_itera5', folder='C:/Users/rodri/Dropbox/JMP/figures/model economy',  *args, **kwargs):
    r"""
    Computes and returns an approximate fixed point of the function `T`.
    The default method `'iteration'` simply iterates the function given
    an initial condition `v` and returns :math:`T^k v` when the
    condition :math:`\lVert T^k v - T^{k-1} v\rVert \leq
    \mathrm{error\_tol}` is satisfied or the number of iterations
    :math:`k` reaches `max_iter`. Provided that `T` is a contraction
    mapping or similar, :math:`T^k v` will be an approximation to the
    fixed point.
    The method `'imitation_game'` uses the "imitation game algorithm"
    developed by McLennan and Tourky [1]_, which internally constructs
    a sequence of two-player games called imitation games and utilizes
    their Nash equilibria, computed by the Lemke-Howson algorithm
    routine. It finds an approximate fixed point of `T`, a point
    :math:`v^*` such that :math:`\lVert T(v) - v\rVert \leq
    \mathrm{error\_tol}`, provided `T` is a function that satisfies the
    assumptions of Brouwer's fixed point theorm, i.e., a continuous
    function that maps a compact and convex set to itself.
    Parameters
    ----------
    T : callable
        A callable object (e.g., function) that acts on v
    v : object
        An object such that T(v) is defined; modified in place if
        `method='iteration' and `v` is an array
    error_tol : scalar(float), optional(default=1e-3)
        Error tolerance
    max_iter : scalar(int), optional(default=50)
        Maximum number of iterations
    verbose : scalar(int), optional(default=2)
        Level of feedback (0 for no output, 1 for warnings only, 2 for
        warning and residual error reports during iteration)
    print_skip : scalar(int), optional(default=5)
        How many iterations to apply between print messages (effective
        only when `verbose=2`)
    method : str, optional(default='iteration')
        str in {'iteration', 'imitation_game'}. Method of computing
        an approximate fixed point
    args, kwargs :
        Other arguments and keyword arguments that are passed directly
        to  the function T each time it is called
    Returns
    -------
    v : object
        The approximate fixed point
    References
    ----------
    .. [1] A. McLennan and R. Tourky, "From Imitation Games to
       Kakutani," 2006.
    """
    if max_iter < 1:
        raise ValueError('max_iter must be a positive integer')

    if verbose not in (0, 1, 2):
        raise ValueError('verbose should be 0, 1 or 2')

    if method not in ['iteration', 'imitation_game']:
        raise ValueError('invalid method')

    

    # method == 'iteration'
    iterate = 0

    if verbose == 2:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    while True:
        v = policies[0]
        
        sol = T(policies, *args, **kwargs)
    
        new_v = sol[0]
        iterate += 1
        error = np.max(np.abs(new_v - v))
        policies = np.copy(sol)
        
        if save==True:
            np.save(folder+save_name,new_v)

        try:
            v[:] = new_v
            
        except TypeError:
            v = new_v

        if error <= error_tol or iterate >= max_iter:
            break
        

        if verbose == 2:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

    if verbose == 2:
        etime = time.time() - start_time
        print_skip = 1
        _print_after_skip(print_skip, iterate, error, etime)
    if verbose >= 1:
        if error > error_tol:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return policies



def compute_fixed_point_pol(T, v, pol, error_tol=1e-3, max_iter=50, verbose=2,
                        print_skip=5, method='iteration', save=True, save_name='v_itera5', folder='C:/Users/rodri/Dropbox/JMP/figures/model economy',  *args, **kwargs):
    r"""
    Computes and returns an approximate fixed point of the function `T`.
    The default method `'iteration'` simply iterates the function given
    an initial condition `v` and returns :math:`T^k v` when the
    condition :math:`\lVert T^k v - T^{k-1} v\rVert \leq
    \mathrm{error\_tol}` is satisfied or the number of iterations
    :math:`k` reaches `max_iter`. Provided that `T` is a contraction
    mapping or similar, :math:`T^k v` will be an approximation to the
    fixed point.
    The method `'imitation_game'` uses the "imitation game algorithm"
    developed by McLennan and Tourky [1]_, which internally constructs
    a sequence of two-player games called imitation games and utilizes
    their Nash equilibria, computed by the Lemke-Howson algorithm
    routine. It finds an approximate fixed point of `T`, a point
    :math:`v^*` such that :math:`\lVert T(v) - v\rVert \leq
    \mathrm{error\_tol}`, provided `T` is a function that satisfies the
    assumptions of Brouwer's fixed point theorm, i.e., a continuous
    function that maps a compact and convex set to itself.
    Parameters
    ----------
    T : callable
        A callable object (e.g., function) that acts on v
    v : object
        An object such that T(v) is defined; modified in place if
        `method='iteration' and `v` is an array
    error_tol : scalar(float), optional(default=1e-3)
        Error tolerance
    max_iter : scalar(int), optional(default=50)
        Maximum number of iterations
    verbose : scalar(int), optional(default=2)
        Level of feedback (0 for no output, 1 for warnings only, 2 for
        warning and residual error reports during iteration)
    print_skip : scalar(int), optional(default=5)
        How many iterations to apply between print messages (effective
        only when `verbose=2`)
    method : str, optional(default='iteration')
        str in {'iteration', 'imitation_game'}. Method of computing
        an approximate fixed point
    args, kwargs :
        Other arguments and keyword arguments that are passed directly
        to  the function T each time it is called
    Returns
    -------
    v : object
        The approximate fixed point
    References
    ----------
    .. [1] A. McLennan and R. Tourky, "From Imitation Games to
       Kakutani," 2006.
    """
    if max_iter < 1:
        raise ValueError('max_iter must be a positive integer')

    if verbose not in (0, 1, 2):
        raise ValueError('verbose should be 0, 1 or 2')

    if method not in ['iteration', 'imitation_game']:
        raise ValueError('invalid method')

    

    # method == 'iteration'
    iterate = 0

    if verbose == 2:
        start_time = time.time()
        _print_after_skip(print_skip, it=None)

    while True:
        new_v,new_pol = T(v,pol, *args, **kwargs)
        
        iterate += 1
        error = np.max(np.abs(new_v - v))
        
        if save==True:
            np.save(folder+save_name,new_v)

        try:
            v[:] = new_v
        except TypeError:
            v = new_v
            pol = new_pol

        if error <= error_tol or iterate >= max_iter:
            break

        if verbose == 2:
            etime = time.time() - start_time
            _print_after_skip(print_skip, iterate, error, etime)

    if verbose == 2:
        etime = time.time() - start_time
        print_skip = 1
        _print_after_skip(print_skip, iterate, error, etime)
    if verbose >= 1:
        if error > error_tol:
            warnings.warn(_non_convergence_msg, RuntimeWarning)
        elif verbose == 2:
            print(_convergence_msg.format(iterate=iterate))

    return v