#!/bin/bash

jupyter nbconvert --to markdown notebooks/README.ipynb
mv notebooks/README.md .
rm -rf README_files
mv notebooks/README_files/ .
