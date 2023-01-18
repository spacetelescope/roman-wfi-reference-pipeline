# Roman WFI Reference File Pipeline

## Installation

It is recommended to begin with a clean environment, such as:
```buildoutcfg
conda create -n wfirefpipe python=3.9
```
To install the package, run:
```buildoutcfg
pip install .
```
This will also install all of the dependencies.

## Documentation

To get the full documentation, install the package as described above, then go to the /src/docs/ directory and run:

```buildoutcfg
make html
```

The documentation can then be found in doc/build/html/index.html.

## Slack Integration

To use the Slack notifications, you must set up a Slack token. 
When you have a token, you can point to it with the environment variable WFI_SLACK_TOKEN.

### Miscellaneous

<div>Project icon made by <a href="" title="Good Ware">Good Ware</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
