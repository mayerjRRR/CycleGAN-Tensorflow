import os
import git
import datetime

MD_LINE_END = "  "

def get_repo_status_string():
    repo = git.Repo(os.getcwd())
    current_branch = repo.head.reference

    branch_name = current_branch.name
    commit_hash = current_branch.commit.hexsha
    commit_message = current_branch.commit.message
    commit_date = datetime.datetime.fromtimestamp(current_branch.commit.committed_date)

    result = ""
    result += "Branch: "+branch_name+MD_LINE_END+os.linesep
    result += f"Latest Commit ({commit_date.strftime('%Y-%m-%d_%H-%M-%S')}): "+commit_hash+MD_LINE_END+os.linesep
    result += commit_message+MD_LINE_END

    return result