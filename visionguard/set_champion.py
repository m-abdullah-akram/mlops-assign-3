"""python -m visionguard.set_champion --version 2   (gives that version the alias 'champion')"""
import argparse

from visionguard import config, registry

ap = argparse.ArgumentParser()
ap.add_argument("--name", default=f"{config.PROJECT_NAME}_model")
ap.add_argument("--version", required=True)
ap.add_argument("--alias", default="champion")
a = ap.parse_args()
registry.set_alias(a.name, a.version, a.alias)
