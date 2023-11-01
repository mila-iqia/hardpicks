#!/bin/sh 

# exit at the first error
set -e

# flake8 linting
flake8 --ignore D,W503 --max-line-length=120 .  # Check everything but docstrings
flake8 --select D --ignore D104,D100,D401 --docstring-convention google --exclude tests/  # Check only the docstrings

# Raise error if any staged notebooks contain outputs
GITDIR=$(git rev-parse --show-toplevel) # Full path to git working directory
IPYNB_FILES=$(git diff --name-only --cached | grep .ipynb || true) # Find all committed notebooks
if [ "$IPYNB_FILES" != "" ] && [ -z $ALLOW_IPYNB ]; then
    for f in $IPYNB_FILES
    do
        DIFF=$(jupyter nbconvert --log-level ERROR --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to notebook --stdout $GITDIR/$f | diff $GITDIR/$f  - || :)
        if [ "$DIFF" != "" ]; then
            echo "
            The notebook $GITDIR/$f contains outputs.
            Remove them all before committing. 
            ***Hint*** use the command:

            jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to notebook --inplace $GITDIR/$f

	    To ignore this error, and add a notebook with outputs, use:

	    export ALLOW_IPYNB=1
            "
            exit 1
        fi
    done
fi
