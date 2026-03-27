import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    theta : float
        Bernoulli parameter, must satisfy 0 < theta < 1.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(theta) + (1-x_i) log(1-theta)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if theta is not in (0,1)
    - Raise ValueError if data contains values other than 0 and 1
    """
    # Check if data is empty
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    # Check if theta is in (0,1)
    if theta <= 0 or theta >= 1:
        raise ValueError("Theta must be in (0,1)")
    
    # Check if data contains only 0 and 1
    # Convert to float to handle numpy types, then check if values are 0 or 1
    if not all(float(x) in [0.0, 1.0] for x in data):
        raise ValueError("Data must contain only 0 and 1")
    
    # Compute log-likelihood
    log_likelihood = sum(x * math.log(theta) + (1 - x) * math.log(1 - theta) for x in data)
    
    return log_likelihood


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    candidate_thetas : array-like or None
        Optional candidate theta values to compare using log-likelihood.
        If None, use [0.2, 0.5, 0.8].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Bernoulli MLE
        - 'num_successes': int
        - 'num_failures': int
        - 'log_likelihoods': dict
            Mapping candidate theta -> log-likelihood
        - 'best_candidate': float
            Candidate theta with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using bernoulli_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    # Validate data
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    # Check if data contains only 0 and 1
    if not all(float(x) in [0.0, 1.0] for x in data):
        raise ValueError("Data must contain only 0 and 1")
    
    # Compute MLE analytically
    num_successes = sum(data)
    num_failures = len(data) - num_successes
    mle = num_successes / len(data) if len(data) > 0 else 0
    
    # Use default candidate_thetas if None
    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]
    
    # Compute log-likelihoods for candidates
    log_likelihoods = {}
    for theta in candidate_thetas:
        try:
            log_likelihoods[theta] = bernoulli_log_likelihood(data, theta)
        except ValueError:
            # Skip invalid theta values (though they should be valid based on requirements)
            log_likelihoods[theta] = -float('inf')
    
    # Find best candidate (first one in case of ties)
    best_candidate = max(candidate_thetas, key=lambda theta: log_likelihoods.get(theta, -float('inf')))
    
    return {
        'mle': mle,
        'num_successes': int(num_successes),
        'num_failures': int(num_failures),
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    lam : float
        Poisson rate, must satisfy lam > 0.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(lam) - lam - log(x_i!)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if lam <= 0
    - Raise ValueError if data contains negative or non-integer values

    Notes
    -----
    You may use math.lgamma(x + 1) for log(x!) since log(x!) = lgamma(x+1).
    """
    # Check if data is empty
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    # Check if lam > 0
    if lam <= 0:
        raise ValueError("Lambda must be > 0")
    
    # Check if data contains only nonnegative integers
    # Convert to float to handle numpy types, then check
    for x in data:
        x_float = float(x)
        if x_float < 0 or x_float != int(x_float):
            raise ValueError("Data must contain only nonnegative integer values")
    
    # Compute log-likelihood
    log_likelihood = sum(x * math.log(lam) - lam - math.lgamma(x + 1) for x in data)
    
    return log_likelihood


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    candidate_lambdas : array-like or None
        Optional candidate lambdas to compare using log-likelihood.
        If None, use [1.0, 3.0, 5.0].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Poisson MLE
        - 'sample_mean': float
        - 'total_count': int
        - 'n': int
        - 'log_likelihoods': dict
            Mapping candidate lambda -> log-likelihood
        - 'best_candidate': float
            Candidate lambda with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using poisson_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    # Validate data
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    # Check if data contains only nonnegative integers
    for x in data:
        x_float = float(x)
        if x_float < 0 or x_float != int(x_float):
            raise ValueError("Data must contain only nonnegative integer values")
    
    # Compute MLE analytically (sample mean)
    n = len(data)
    total_count = sum(data)
    sample_mean = total_count / n
    mle = sample_mean
    
    # Use default candidate_lambdas if None
    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]
    
    # Compute log-likelihoods for candidates
    log_likelihoods = {}
    for lam in candidate_lambdas:
        try:
            log_likelihoods[lam] = poisson_log_likelihood(data, lam)
        except ValueError:
            # Skip invalid lambda values (though they should be valid based on requirements)
            log_likelihoods[lam] = -float('inf')
    
    # Find best candidate (first one in case of ties)
    best_candidate = max(candidate_lambdas, key=lambda lam: log_likelihoods.get(lam, -float('inf')))
    
    return {
        'mle': mle,
        'sample_mean': sample_mean,
        'total_count': int(total_count),
        'n': int(n),
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }
