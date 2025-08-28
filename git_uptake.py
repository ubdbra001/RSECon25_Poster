from github import Github, Auth
import numpy as np
from dotenv import load_dotenv
from collections import OrderedDict
import os
import hashlib


def load_api_key(key_name: str = "API_TOKEN"):
    load_dotenv()
    api_key = os.getenv(key_name)

    return api_key

# Method for auth
def github_connect(api_token: str) -> Github:
    auth = Auth.Token(api_token)
    githubObj = Github(auth=auth)

    return githubObj

def get_pull_requests(githubObj: Github, repo_name: str, state: str = "all") -> list:
    repo = githubObj.get_repo(repo_name)
    pr_list = list(repo.get_pulls(state=state))

    return pr_list

def get_contributors(githubObj: Github, repo_name: str):
    repo = githubObj.get_repo(repo_name)
    contributors = [contributor for contributor in repo.get_contributors()]

    return contributors

def get_trainees(cge_pulls, exclude_list):
    trainee_dict = {}
    
    # For each pull
    for pull in cge_pulls:
    
        # Get the user who created it
        user = pull.user
    
        # if the user is in the exclude list then move on
        if user in exclude_list:
            continue
    
        # Get the date of PR creation
        pr_date = pull.created_at.date()
    
        # If the user isn't already in the list of trainees then add them
        if user not in trainee_dict.keys():
            trainee_dict.update({user: pr_date})
        else:
        # If they are then see if the PRs are on the same day, if not remove
        # the user form the trainees list and add them to the exclude list
            if trainee_dict[user] != pr_date:
                del trainee_dict[user]
                exclude_list.add(user)

    trainee_list = [OrderedDict({"trainee": trainee, "PR_date": PR_date}) 
                    for trainee, PR_date in trainee_dict.items()]

    return trainee_list

def get_trainee_info(trainee_item):
    trainee, PR_date = trainee_item.values()
    acc_creation_delta = trainee.created_at.date() - PR_date
    trainee_item["account_creation"] = acc_creation_delta.days

    repo_info_dict = get_trainee_repo_info(trainee, PR_date)

    trainee_item.update(repo_info_dict)

    anaonymise_trainee_data(trainee_item)

def get_trainee_repo_info(trainee, PR_date):
    repo_info = {
        "number_commits": [],
        "days_alive": [],
        "creation_delta": []
    }

    for repo in trainee.get_repos():
        if (not repo.fork) and (trainee in repo.get_contributors()):
            commit_list = list(repo.get_commits())
            repo_info["number_commits"].append(len(commit_list))

            newest_commit = commit_list[0]
            oldest_commit = commit_list[-1]

            days_alive = newest_commit.commit.committer.date.date() - oldest_commit.commit.committer.date.date()
            repo_info["days_alive"].append(days_alive.days)

            creation_delta = oldest_commit.commit.committer.date.date() - PR_date
            repo_info["creation_delta"].append(creation_delta.days)
    
    return repo_info

def anaonymise_trainee_data(trainee_item):
    h = hashlib.new('sha256')
    h.update(trainee_item["trainee"].login.encode())
    trainee_item["trainee"] = h.hexdigest()

    del trainee_item["PR_date"]
