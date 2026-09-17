"""Exact linear conditional fixture for temporal model regressions."""
import itertools

def analytic_case(intercept=3.0, root_mean=1.0):
    """X_t=mu+eX_t; Y0=c+2X0+.5eY0; Y1=c+2X1+.5Y0+.5eY1.

    All e independent N(0,1); Z_t=eZ_t is an irrelevant manipulable root.
    The conditional mean and variance at fixed (X1=x,Y0=b) are c+2x+.5b
    and .25. Separate regressions on X1 and Y0 double-count marginal means
    and count unconditioned parent variation as conditional noise.
    """
    mean_y0 = intercept + 2 * root_mean
    mean_y1 = intercept + 2 * root_mean + 0.5 * mean_y0
    return dict(intercept=intercept, root_mean=root_mean,
                mean_y0=mean_y0, mean_y1=mean_y1,
                var_y0=4.25, true_conditional_variance=0.25,
                summed_conditional_variance=5.5625,
                expected_mean_bias=mean_y1)


def moment_data(case):
    """Degree-two Gaussian moment cubature, not a purported observational run."""
    import numpy as np
    e = np.asarray(list(itertools.product((-1.0, 1.0), repeat=6)))
    x = case['root_mean'] + e[:, [0, 3]]
    z = e[:, [1, 4]]
    y0 = case['intercept'] + 2 * x[:, 0] + 0.5 * e[:, 2]
    y1 = case['intercept'] + 2 * x[:, 1] + 0.5 * y0 + 0.5 * e[:, 5]
    return {'X': x, 'Z': z, 'Y': np.column_stack((y0, y1))}


class LinearProjection:
    """Population linear conditional moments on the moment-exact design."""
    def __init__(self, x, y):
        import numpy as np
        x, y = np.asarray(x), np.asarray(y)
        design = np.column_stack((np.ones(len(x)), x))
        self.coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        self.variance = np.mean((y - design @ self.coefficients) ** 2, axis=0)

    def predict(self, x):
        import numpy as np
        x = np.atleast_2d(x)
        design = np.column_stack((np.ones(len(x)), x))
        return design @ self.coefficients, np.tile(self.variance, (len(x), 1))
