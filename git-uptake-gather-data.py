import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import git_uptake as gu
    import marimo as mo

    import pickle
    import datetime

    from copy import deepcopy, copy
    return copy, deepcopy, gu, mo, pickle


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Connect to GH API

    Sart by loading in the API key and creating a object that sends requests and receives responses from the Github API
    """
    )
    return


@app.cell
def _(gu):
    key = gu.load_api_key()
    return (key,)


@app.cell
def _(gu, key):
    gh_obj = gu.github_connect(key)
    return (gh_obj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Get pull requests from the `collaborative_github_exercise` repository""")
    return


@app.cell
def _(gh_obj, gu):
    # Get all the pulls and contributors to the collaborative github exercise (cge)
    repo_name = "RSE-Sheffield/collaborative_github_exercise"
    cge_pulls = gu.get_pull_requests(gh_obj, repo_name, state="all")
    return (cge_pulls,)


@app.cell
def _(cge_pulls):
    unique_users = {pull.user for pull in cge_pulls}
    print(f"Total number of unique users: {len(unique_users)}")
    return (unique_users,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Remove non-trainee contributors to the repo

    1. Remove any contributor who has also contributed to the (old and new) RSE website (and therefore is a member of RSE staff)
    2. Remove any contributor who contibuted multiple pull requests on different days
    """
    )
    return


@app.cell
def _(gh_obj):
    print(f"Requests remaining: {gh_obj.get_rate_limit().core.remaining}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Get the list of RSE staff
    It won't change very often/at all, so we can cache it
    """
    )
    return


@app.cell
def _(gu, pickle):
    def get_rse_staff(gh_obj, file_out):
        rse_website_repo = "RSE-Sheffield/RSE-Sheffield.github.io"
        rse_old_website_repo = "RSE-Sheffield/old_team_site_nikola"
        newer_rse_staff = gu.get_contributors(gh_obj, rse_website_repo)
        older_rse_staff = gu.get_contributors(gh_obj, rse_old_website_repo)
        rse_staff = set(newer_rse_staff + older_rse_staff)

        with open(file_out, "wb") as staff_file:
            pickle.dump(rse_staff, staff_file)

        return rse_staff
    return (get_rse_staff,)


@app.cell
def _(get_rse_staff, gh_obj, pickle):
    rse_staff_file = "rse_staff.pkl"
    force = False

    if force:
        rse_staff = get_rse_staff(gh_obj, rse_staff_file)
    else:
        try:
            with open(rse_staff_file, "rb") as staff_file:
                rse_staff = pickle.load(staff_file)
        except FileNotFoundError:
            rse_staff = get_rse_staff(gh_obj, rse_staff_file)

    print(f"Number of RSE staff: {len(rse_staff)}")
    return (rse_staff,)


@app.cell
def _(cge_pulls, rse_staff, unique_users):
    rse_contributors = unique_users.intersection(rse_staff)
    print(f"RSE staff contributors: {len(rse_contributors)}")
    rse_pulls = [pull for pull in cge_pulls if pull.user in rse_contributors]
    print(f"RSE staff PRs: {len(rse_pulls)}")
    return


@app.cell
def _(cge_pulls, rse_staff, unique_users):
    print(f"Unique users minus RSE staff: {len(unique_users - rse_staff)}")
    non_rse_pulls = [pull for pull in cge_pulls if pull.user not in rse_staff]
    print(f"PRs minus RSE staff PRs: {len(non_rse_pulls)}")
    return


@app.cell
def _(gh_obj):
    print(f"Requests remaining: {gh_obj.get_rate_limit().core.remaining}")
    return


@app.cell
def _(copy, rse_staff):
    # Remove all RSE staff from the list of cge contributors to get a list of trainees
    exclude_list = copy(rse_staff)
    return (exclude_list,)


@app.cell
def _(cge_pulls, exclude_list, gu):
    trainees = gu.get_trainees(cge_pulls, exclude_list)
    return (trainees,)


@app.cell
def _(gh_obj):
    print(f"Requests remaining: {gh_obj.get_rate_limit().core.remaining}")
    return


@app.cell
def _(cge_pulls, exclude_list, rse_staff):
    other_non_trainees = exclude_list - rse_staff
    print(f"Other potential non-trainees: {len(other_non_trainees)}")
    non_trainees_pulls = [pull for pull in cge_pulls if pull.user in other_non_trainees]
    print(f"Other potential non-trainees PRs: {len(non_trainees_pulls)}")

    return


@app.cell
def _(exclude_list, unique_users):
    print(f"Probably Trainees: {len(unique_users - exclude_list)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Next steps:
    - Show frequency and attendance of lessons (N pull requests over time)
    - What is the distribution of account creation dates
        - What proportion of trainees created their account more than a month before the training
        - What proportion of accounts had commits/contributions to their accounts before the training
    - What is the proportion of accounts with activity after the training (same criteria as before training)
        - What is the distribution of this activity?
    - What proportion of the trainees created a branch or opened a PR pre vs post training
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Pull requests over time
    Limited to trainees only
    """
    )
    return


@app.cell
def _(pickle, trainees):
    PR_creation_dates = [item["PR_date"] for item in trainees]
    with open("PR_creation_dates.pkl", "wb") as file:
        pickle.dump(PR_creation_dates, file)
    return


@app.cell
def _():
    from requests.exceptions import RetryError
    return (RetryError,)


@app.cell
def _(RetryError, deepcopy, gu, trainees):
    copy_trainees = deepcopy(trainees)

    failed_requests = []

    for index, item in enumerate(copy_trainees):
        print(f"Processing: {index} - {item['trainee'].login}")
        try:
            gu.get_trainee_info(item)
        except RetryError as e:
            failed_requests.append(index)
            print(e)
    return copy_trainees, failed_requests


@app.cell
def _(gh_obj):
    print(f"Requests remaining: {gh_obj.get_rate_limit().core.remaining}")
    return


@app.cell
def _(copy_trainees):
    copy_trainees
    return


@app.cell
def _(failed_requests):
    failed_requests
    return


@app.cell
def _(copy_trainees, pickle):
    with open("gh_trainees_data.pkl", "wb") as trainees_file:
        pickle.dump(copy_trainees, trainees_file, protocol=pickle.HIGHEST_PROTOCOL)
    return


@app.cell
def _(copy_trainees):
    len(copy_trainees)
    return


if __name__ == "__main__":
    app.run()
