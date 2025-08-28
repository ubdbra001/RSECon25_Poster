import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import datetime
    import pickle
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter, MultipleLocator
    from pathlib import Path
    return (
        MultipleLocator,
        PercentFormatter,
        datetime,
        mdates,
        np,
        pickle,
        pl,
        plt,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Load data for plotting
    - Pull request creation dates
    - Trainee data
    """
    )
    return


@app.cell
def _(pickle):
    PR_creation_date_file = "PR_creation_dates.pkl"

    with open(PR_creation_date_file, "rb") as PR_date_file:
        PR_creation_dates = pickle.load(PR_date_file)
    return (PR_creation_dates,)


@app.cell
def _(pickle):
    trainee_data_file = "gh_trainees_data.pkl"

    with open(trainee_data_file, "rb") as trainee_file:
        trainee_data = pickle.load(trainee_file)

    trainee_data
    return (trainee_data,)


@app.cell
def _(plt):
    plt.style.use('default')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Plots needed:

    1. PR creation date over time
    2. Account creation relative to PR
    3. Proportion of attendees with gh activity after PR (over time) - For all
    4. Divide the above:  
       a. Same but only for those with no/little activity before PR  
       b. For those with activity before PR does it increase? (more repos, more commits, etc)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## PR creation date over time""")
    return


@app.cell
def _(datetime, pl):
    oldest_date = datetime.date(2016,1,1)
    now = datetime.datetime.now().date()
    date_bins = pl.date_range(oldest_date, now, "3mo", eager=True).to_list()
    return (date_bins,)


@app.cell
def _(PR_creation_dates, date_bins, mdates, plt):
    PR_time_fig, PR_time_ax = plt.subplots()
    PR_time_ax.hist(PR_creation_dates, bins=date_bins, rwidth=0.8)
    PR_time_ax.set_ylabel(r'Number of Trainees')
    PR_time_ax.set_xlabel(r'Date (3 month summary)')
    PR_time_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    PR_time_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1)))
    PR_time_ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4,7,10)))
    PR_time_ax.tick_params(axis="x", which="major", length = 7)
    PR_time_ax.grid(axis="both", ls=":")
    PR_time_ax.grid(True)
    PR_time_fig.autofmt_xdate()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Account creation relative to PR

    Is skewed, will probably need to include inset for bar nearest 0 to expand it.  
    i.e. Most accounts (~40%) created within 30 days of training
    """
    )
    return


@app.cell
def _(trainee_data):
    acc_creation_days = [item.get("account_creation") for item in trainee_data]
    acc_creation_days = [abs(value) for value in acc_creation_days if value is not None]
    return (acc_creation_days,)


@app.cell
def _(acc_creation_days):
    len([value for value in acc_creation_days if value < 30])/len(acc_creation_days)
    return


@app.cell
def _(acc_creation_days, np):
    acc_creation_weights = np.ones(len(acc_creation_days)) / len(acc_creation_days)
    return (acc_creation_weights,)


@app.cell
def _():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    return inset_axes, mark_inset


@app.cell
def _(
    MultipleLocator,
    PercentFormatter,
    acc_creation_days,
    acc_creation_weights,
    inset_axes,
    mark_inset,
    plt,
):
    acc_time_fig, acc_time_ax = plt.subplots()
    acc_time_ax.hist(acc_creation_days,
                     rwidth=0.9,
                     bins = 25,
                     weights=acc_creation_weights,
                     range=(0,4500),
                     zorder=10)
    acc_time_ax.set_ylabel(r'Percent of Trainees')
    acc_time_ax.set_xlabel(r'Days')
    acc_time_ax.set_title(r'Time between account creation and training', pad=20)
    acc_time_ax.yaxis.set_major_formatter(PercentFormatter(1))
    acc_time_ax.xaxis.set_major_locator(MultipleLocator(720))
    acc_time_ax.xaxis.set_minor_locator(MultipleLocator(180))
    acc_time_ax.set_ylim(top=0.6)
    acc_time_ax.grid(axis="both", ls=":")
    acc_time_ax.grid(zorder = 0)

    acc_time_ax_inset = inset_axes(acc_time_ax,
                                   width="50%",
                                   height="50%",
                                   loc='right',
                                   bbox_to_anchor=(0, 0.05, 0.95, 0.95),
                                   bbox_transform=acc_time_ax.transAxes)

    _, _, patches = acc_time_ax_inset.hist(acc_creation_days,
                                           rwidth=0.7,
                                           bins = 6,
                                           weights=acc_creation_weights,
                                           range=(0,180),
                                           zorder = 10)
    patches[0].set_facecolor('#ff7f0e')
    acc_time_ax_inset.yaxis.set_major_formatter(PercentFormatter(1))
    acc_time_ax_inset.xaxis.set_major_locator(MultipleLocator(30))
    acc_time_ax_inset.grid(axis="both", ls=":")
    acc_time_ax_inset.grid(zorder = 0)
    acc_time_ax_inset.set_ylim(top=0.5)

    mark_inset(acc_time_ax, acc_time_ax_inset, loc1=2, loc2=4, ec='0.5', ls=':', zorder=0)

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Proportion of trainees with GH activity post training
    i.e. how long does it take for a proportion of trainees to use git/github after the training?
    """
    )
    return


@app.cell
def _():
    new_user_cutoff = -30
    days_post_PR = 3
    days_pre_PR = -3
    return days_post_PR, days_pre_PR, new_user_cutoff


@app.cell
def _(days_post_PR, np, trainee_data):

    all_days = [np.array(item.get("creation_delta")) for item in trainee_data if item.get("creation_delta") is not None]
    all_trainees = [days.min(initial = 1_000_000, where = days >= days_post_PR) if np.any(days >= days_post_PR) else np.nan for days in all_days]

    all_trainees_weights = np.ones(len(all_trainees)) / len(all_trainees)

    all_trainees_vals = [value for value in all_trainees if not np.isnan(value)]
    all_trainees_vals_year = [value for value in all_trainees_vals if value <= 360]

    print(f"Trainees with new activity post training: {len(all_trainees_vals)} (proportion: {len(all_trainees_vals)/len(trainee_data):.3f})")
    print(f"Trainees with new activity post training within year: {len(all_trainees_vals_year)} (proportion: {len(all_trainees_vals_year)/len(trainee_data):.3f})")
    return all_trainees, all_trainees_weights


@app.cell
def _(all_trainees, days_post_PR, new_user_cutoff, np, trainee_data):
    new_users_all_days = [np.array(item["creation_delta"]) for item in trainee_data if item.get("account_creation") is not None and item.get("account_creation") >= new_user_cutoff]

    new_users = [days.min(initial = 1_000_000, where = days >= days_post_PR) if np.any(days >= days_post_PR) else np.nan for days in new_users_all_days]

    new_users_weights = np.ones(len(new_users)) / len(all_trainees)

    new_users_vals = [value for value in new_users if not np.isnan(value)]
    new_users_vals_year = [value for value in new_users_vals if value <= 360]

    print(f"New users with new activity post training: {len(new_users_vals)} (proportion: {len(new_users_vals)/len(trainee_data):.3f})")
    print(f"New users with new activity post training within year: {len(new_users_vals_year)} (proportion: {len(new_users_vals_year)/len(trainee_data):.3f})")
    return new_users, new_users_weights


@app.cell
def _(
    all_trainees,
    days_post_PR,
    days_pre_PR,
    new_user_cutoff,
    np,
    trainee_data,
):
    old_users = [trainee for trainee in trainee_data if trainee.get("account_creation") < new_user_cutoff]

    old_users_wo_activity = [np.array(trainee["creation_delta"]) for trainee in old_users if not np.any(np.array(trainee["creation_delta"]) < days_pre_PR)]

    old_users_new_activity = [days.min(initial = 1_000_000, where = days >= days_post_PR) if np.any(days >= days_post_PR) else np.nan for days in old_users_wo_activity]

    old_users_weights = np.ones(len(old_users_new_activity)) / len(all_trainees)

    old_users_vals = [value for value in old_users_new_activity if not np.isnan(value)]
    old_users_vals_year = [value for value in old_users_vals if value <= 360]


    print(f"Old users: {len(old_users)} (proportion: {len(old_users)/len(trainee_data):.3f})")
    print(f"Old users (no previous activity): {len(old_users_wo_activity)} (proportion: {len(old_users_wo_activity)/len(old_users):.3f})")
    print(f"Old users (no previous activity) with new activity post training: {len(old_users_vals)} (proportion: {len(old_users_vals)/len(trainee_data):.3f})")
    print(f"Old users (no previous activity) with new activity post training within year: {len(old_users_vals_year)} (proportion: {len(old_users_vals_year)/len(trainee_data):.3f})")
    return old_users, old_users_new_activity, old_users_weights


@app.cell
def _(
    all_trainees,
    all_trainees_weights,
    new_users,
    new_users_weights,
    old_users_new_activity,
    old_users_weights,
):
    data = [all_trainees, new_users, old_users_new_activity]
    weights = [all_trainees_weights, new_users_weights, old_users_weights]
    labels = ["All Trainees", "New Users", "Old users with no previous activity"]

    return data, labels, weights


@app.cell
def _(mo):
    mo.md(r"""Cumulative plot showing the days to next commit post PR (over year)""")
    return


@app.cell
def _(MultipleLocator, PercentFormatter, data, labels, plt, weights):
    activity_fig, activity_ax = plt.subplots()
    activity_ax.grid(axis="both", ls=":")
    activity_ax.grid(axis="x", which="minor", ls=":")
    activity_ax.grid(True)

    activity_ax.hist(data, weights = weights, label = labels,
                     bins = 360, range = (0,400), cumulative=True, histtype="step", lw = 2)
    activity_ax.yaxis.set_major_formatter(PercentFormatter(1))
    activity_ax.xaxis.set_major_locator(MultipleLocator(60))
    activity_ax.xaxis.set_minor_locator(MultipleLocator(30))
    activity_ax.set_xlim(right = 360)
    activity_ax.set_ylabel(r'Percent of all Trainees')
    activity_ax.set_xlabel(r'Days after Training')
    activity_ax.set_title(r'Time to next commit post-training', pad=20)
    activity_ax.legend(reverse = True)

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""Cumulative plot showing the days to next commit post PR (over 6 years)""")
    return


@app.cell
def _(MultipleLocator, PercentFormatter, data, plt, weights):
    activity_fig2, activity_ax2 = plt.subplots()
    activity_ax2.grid(axis="both", ls=":")
    activity_ax2.grid(True)
    activity_ax2.hist(data, bins = 360, weights=weights, cumulative=True, histtype="step")
    activity_ax2.yaxis.set_major_formatter(PercentFormatter(1))
    activity_ax2.xaxis.set_major_locator(MultipleLocator(360))
    activity_ax2.xaxis.set_major_locator(MultipleLocator(360))
    activity_ax2.set_xlim(right = 2160)
    activity_ax2.set_ylabel(r'Percent of Trainees')
    activity_ax2.set_xlabel(r'Days after Training')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""Of the 'old users' with previous activity, did activity increase?""")
    return


@app.cell
def _(days_pre_PR, np, old_users):
    old_users_w_activity = [trainee for trainee in old_users if np.any(np.array(trainee["creation_delta"]) < days_pre_PR)]

    return (old_users_w_activity,)


@app.cell
def _(np, old_users_w_activity):
    activity_dict = {
        "repos_before": [],
        "repos_after": [],
        "avg_commits_before": [],
        "avg_commits_after": []
    }


    for trainee in old_users_w_activity:
        creation_delta = np.array(trainee["creation_delta"])
        number_commits = np.array(trainee["number_commits"])
        days_alive = np.array(trainee["days_alive"])

        pre_training_repos = np.logical_and(creation_delta < -3, creation_delta > -360)  

        if np.any(pre_training_repos):
            activity_dict["repos_before"].append(sum(pre_training_repos))
            pre_training_commit_freq = number_commits[pre_training_repos]/ (days_alive[pre_training_repos]+1)
            activity_dict["avg_commits_before"].append(np.mean(pre_training_commit_freq))
        else:
            activity_dict["repos_before"].append(0)
            activity_dict["avg_commits_before"].append(0)

        post_training_repos = np.logical_and(creation_delta > 3, creation_delta < 360)

        if np.any(post_training_repos):
            activity_dict["repos_after"].append(sum(post_training_repos))
            post_training_commit_freq = number_commits[post_training_repos]/(days_alive[post_training_repos]+1)
            activity_dict["avg_commits_after"].append(np.mean(post_training_commit_freq))
        else:
            activity_dict["repos_after"].append(0)
            activity_dict["avg_commits_after"].append(0)

    activity_dict

    return (activity_dict,)


@app.cell
def _(activity_dict, np):
    print(f"Number repos created before: {np.median(activity_dict['repos_before']):.3f}")
    print(f"Number repos created after: {np.median(activity_dict['repos_after']):.3f}")
    print(f"Avg number commits before: {np.median(activity_dict['avg_commits_before']):.3f}")
    print(f"Avg number commits after: {np.median(activity_dict['avg_commits_after']):.3f}")
    return


@app.cell
def _():
    import matplotlib.patches as mpatches 
    return (mpatches,)


@app.cell
def _(activity_dict, mpatches, plt):


    activity_fig3, activity_ax3 = plt.subplots(ncols=2)
    repos_before_plot = activity_ax3[0].violinplot(activity_dict['repos_before'], side = 'low', widths = 0.2, showmeans = True)
    repos_after_plot = activity_ax3[0].violinplot(activity_dict['repos_after'], side = 'high', widths = 0.2,showmeans = True)
    activity_ax3[0].set_ylabel("Number of repositories created", labelpad = 10)
    activity_ax3[0].tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    colours = [mpatches.Patch(color='tab:red', label='Pre-training'),
               mpatches.Patch(color='tab:cyan', label='Post-training')]

    for idx, part in enumerate([repos_before_plot, repos_after_plot]):
        for pc in part['bodies']:
            pc.set_facecolor(colours[idx].get_facecolor())
            pc.set_edgecolor(None)
            pc.set_alpha(0.7)

        part["cbars"].set_color('white')
        part["cbars"].set_alpha(0.2)
        part["cmeans"].set_edgecolor("black")

        for line in ['cmins', 'cmaxes']:
            part[line].set_color("black")
            part[line].set_linestyle(":")
    
    commits_before_plot = activity_ax3[1].violinplot(activity_dict['avg_commits_before'], side = 'low', showmeans=True, widths = 0.2)
    commits_after_plot = activity_ax3[1].violinplot(activity_dict['avg_commits_after'], side = 'high', showmeans=True, widths = 0.2)
    activity_ax3[1].set_ylabel("Mean commit frequency(commits/day)", labelpad = 10)
    activity_ax3[1].tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    for idx, part in enumerate([commits_before_plot, commits_after_plot]):
        for pc in part['bodies']:
            pc.set_facecolor(colours[idx].get_facecolor())
            pc.set_edgecolor(None)
            pc.set_alpha(0.7)

        part["cbars"].set_color('white')
        part["cbars"].set_alpha(0.2)
        part["cmeans"].set_edgecolor("black")

        for line in ['cmins', 'cmaxes']:
            part[line].set_edgecolor("black")
            part[line].set_linestyle(":")

    activity_fig3.suptitle("Changes in activity for existing users (1 year pre-/post- training)")
    activity_fig3.tight_layout()

    # Create a single legend for the entire figure
    activity_fig3.legend(handles=colours, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncols= 2)


    plt.show()
    return


@app.cell
def _(mpatches):
    colours2 = [mpatches.Patch(color='tab:red', label='Pre-training'),
               mpatches.Patch(color='tab:cyan', label='Post-training')]
    return (colours2,)


@app.cell
def _(colours2):
    colours2[0].get_facecolor()
    return


if __name__ == "__main__":
    app.run()
