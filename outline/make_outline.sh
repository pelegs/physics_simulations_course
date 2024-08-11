#!/usr/bin/env bash

pandoc outline.md --number-sections --standalone -V "date:$(date '+%d-%m-%Y')" --output=outline.html
