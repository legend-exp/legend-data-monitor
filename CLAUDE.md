## General code guidelines

Code changes that are short and targeted are generally prefered.
Any code comments inline should fit on the line next to the code they refer to.
Avoid where possible long code comments, the code itself should be readable 
and understandable in a way that these are not necessary.
Docstrings should follow numpy convention e.g.

def func(a,b);
    """
    this func does something

    params
    ------
    a : str
        a is a string
    b: float
        b is a float

    returns
    -------
    var: int
        this func returns an int

## Github guidelines

Always run pre-commit before committing with pre-commit run -a
Try to keep commit messages short and to the point, likewise with pr body and comments.
Avoid too many commits, keep the git history as clean as possible. 

## repo specific

This code is designed to be used by a monitoring dashboard (https://github.com/legend-exp/legend-monitor-dashboard) run in panel, outputs 
should therefore be as lightweight as possible to ensure this dashboard is fast.
It also runs on a shared machine so resource usage should be as low as possible.
Keep these in mind when making changes.
The code also interacts with auto-giorgio (https://github.com/ggmarshall/auto-giorgio), a bot using claude to diagnose issues, warnings or flags for detector 
instabilities should be clear to make it easy for this bot to diagnose.