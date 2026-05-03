from scipy.stats import t as tstats
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count

################
# Constants
#################

community_names = {
    0: "Math & Computing",
    1: "Phys & Eng",
    2: "Life & Earth",
    3: "Bio & Health",
    4: "Social Sciences",
    # 5: "Humanities",
}

# define colors for communities
community_colors = {
    0: "rgb(38, 96, 164)",  # dark blue
    1: "rgb(0, 128, 128)",  # light brown
    2: "rgb(127, 200, 248)",  # light blue
    3: "rgb(167, 38, 8)",  # red
    4: "rgb(241, 153, 83)",  # melon
    5: "rgb(247, 203, 21)",  # teal
    6: "rgb(255, 102, 0)",  # orange
    7: "rgb(0, 153, 153)",  # blue green
    8: "rgb(255, 153, 51)",  # orange
    9: "rgb(0, 102, 204)",  # blue
    10: "rgb(255, 204, 0)",  # yellow
    11: "rgb(204, 0, 0)",  # red
}

# define colors for gender
gender_colors = {
    "F": "rgb(241,126,108)",
    "M": "rgb(92,135,151)",
    "Woman": "rgb(241,126,108)",
    "Man": "rgb(92,135,151)",
}

################
# Functions
#################


# confidence interval
def calculate_confidence_interval(group):
    std = group.std()
    n = len(group)
    conf_level = 0.95
    dof = n - 1
    t_value = tstats.ppf((1 + conf_level) / 2, dof)
    se = std / np.sqrt(n)
    margin_error = t_value * se
    return margin_error


# rgb to rgba
def rgb2rgba(rgb, alpha):
    # extract number by regex
    rgb = re.findall(r"\d+", rgb)
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


# generate n gradient colors
def generate_gradient_colors(n, palette_name="viridis_r"):
    # Define a Seaborn color palette (you can choose any color map)
    palette = sns.color_palette(
        palette_name, n
    )  # Generates 4 evenly spaced colors from the 'viridis' color map

    # Convert the palette colors to rgb(x, y, z) format
    rgb_colors = [
        "rgb({}, {}, {})".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in palette
    ]

    # Display the colors
    return rgb_colors


# Replicate Seaborn's regplot with confidence intervals and a LOWESS fit
def bootstrap_lowess(args):
    lowess_y, lowess_x, residuals, frac, seed = args
    np.random.seed(seed)
    resample = np.random.choice(residuals, size=len(residuals), replace=True)
    return sm.nonparametric.lowess(lowess_y + resample, lowess_x, frac=frac)[:, 1]


def get_lowess_confidence_interval(x, y, frac=0.5, n_boot=100, conf_level=0.95):
    # Fit LOWESS
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    lowess_x, lowess_y = lowess.T

    # Calculate residuals
    residuals = y - np.interp(x, lowess_x, lowess_y)

    # Prepare arguments for parallel processing
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for bootstrapping")
    args = [(lowess_y, lowess_x, residuals, frac, i) for i in range(n_boot)]

    # Parallelize bootstrap process using Pool
    with Pool(num_cores) as pool:
        bootstrapped = np.array(pool.map(bootstrap_lowess, args))

    # Compute confidence intervals
    lower, upper = np.percentile(
        bootstrapped, [(1 - conf_level) * 50, (1 + conf_level) * 50], axis=0
    )

    return lowess_x, lowess_y, lower, upper


# Function to apply LOWESS and calculate confidence intervals
def apply_lowess(group, xcol, ycol, **kwargs):
    x = group[xcol]
    y = group[ycol]

    # Apply LOWESS smoothing
    lowess_x, lowess_y, lower, upper = get_lowess_confidence_interval(x, y, **kwargs)

    # to a pandas
    df = pd.DataFrame(
        {xcol: lowess_x, ycol: lowess_y, "lower": lower, "upper": upper}
    ).drop_duplicates()

    return df


# Function to apply Welch's t-test
def apply_welch_ttest(group, compare_col="gender", target_col="rao"):
    # get unique values of compare_col
    compare_values = group[compare_col].unique()
    if len(compare_values) != 2:
        # throw error if there are not exactly 2 unique values
        raise ValueError(
            f"Expected exactly 2 unique values in {compare_col}, got {len(compare_values)}"
        )

    # Separate the 'rao' scores by gender
    male = group[group[compare_col] == compare_values[0]][target_col]
    female = group[group[compare_col] == compare_values[1]][target_col]

    # Perform Welch's t-test (equal_var=False indicates Welch's test)
    t_stat, p_value = ttest_ind(male, female, equal_var=False)

    # mark stars
    star = (
        "***"
        if p_value < 0.001
        else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    )

    # Return results as a dictionary
    return pd.Series(
        {
            "t_stat": t_stat,
            "p_value": p_value,
            "star": star,
        }
    )


# plotly add lines with error band
def add_lines_with_errorband(
    fig,
    x,
    y,
    upper,
    lower,
    name,
    color,
    showlegend=False,
    showband=True,
    dash=None,
    line_alpha=0.6,
    line_width=1,
    band_alpha=0.2,
    **kwargs,
):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=rgb2rgba(color, line_alpha), width=line_width, dash=dash),
            showlegend=showlegend,
        ),
        **kwargs,
    )

    # add error band
    if showband:
        x_error = np.concatenate((x, x[::-1]))
        y_error = np.concatenate((lower, upper[::-1]))
        fig.add_trace(
            go.Scatter(
                x=x_error,
                y=y_error,
                fill="toself",
                fillcolor=rgb2rgba(color, band_alpha),
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            ),
            **kwargs,
        )

    return fig
