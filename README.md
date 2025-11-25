# Roman WFI Reference File Pipeline

> [!CAUTION]
> The code in this repository is under heavy development and not currently intended for widespread use. 
>
> If you have questions, please create a new issue.


## Installation

It is recommended to begin with a clean environment, such as:
```buildoutcfg
conda create -n wfirefpipe python=3.12
```
To install the package for general usage:
```buildoutcfg
pip install .
```
OR
To install the package for development and testing:
```buildoutcfg
pip install -e .[docs,test]
```
This will also install all of the dependencies.

> NOTE: Installing this way uses the dependencies outlined in `pyproject.toml`.
> These dependencies mirror `requirements.txt`.
> If you would like to install with custom dependencies,
> just update them in requirements.txt and `pip install -r requrements.txt`

Users must create a config.yml living at:
```/wfi_reference_pipeline/src/wfi_reference_pipeline/config/config.yml```
Use example_config.yml in the same directory as a template.

## Contributing
> [!WARNING]
> We are not currently accepting external Pull Requests. However, we plan to
> use the guidelines below in the near future.

To contribute to this project, please become familiar with our [Contributing Guide](https://github.com/spacetelescope/roman-wfi-reference-pipeline/blob/main/CONTRIBUTING.md)

## Documentation

To get the full documentation, install the package as described above, then go to the /src/docs/ directory and run:

```buildoutcfg
make html
```

The documentation can then be found in doc/build/html/index.html.

## Slack Integration

To use the Slack notifications, you must set up a Slack token.
When you have a token, you can point to it with the environment variable WFI_SLACK_TOKEN.

## Updating reference files on CRDS

Go to your .bash_profile and set environment variables to determine the crds state.

This is for the Roman test instance of CRDS.
```buildoutcfg
export CRDS_SERVER_URL="https://roman-crds-test.stsci.edu"
export CRDS_PATH="<local_path_to_cache>/test_crds/"
```
This is for the Roman TVAC CRDS server which is temporary for analysis up to commissioning.
```buildoutcfg
export CRDS_SERVER_URL="https://roman-crds-tvac.stsci.edu"
export CRDS_PATH="<local_path_to_cache>/tvac_crds"
```

Start with a clean environment where you will get the latest versions of romancal, roman attribute
dictionary, and roman data models.
```buildoutcfg
conda create -yn devRFP_RGCenv_update_refs ipython
conda activate devRFP_RGCenv_update_refs
pip install romancal crds
```
By doing this pip install you will get the latest released versions since this workflow is setup
to replace old reference files with the most current.

Now sync crds to get all of the mappings updated.
```buildoutcfg
crds sync --all
```
