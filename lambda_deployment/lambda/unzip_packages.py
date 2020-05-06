# Python Built-Ins:
import os
import shutil
import sys
import zipfile

print("Installing compressed dependencies to /tmp/...")
sys.path.append(os.path.realpath("packages"))
for package in os.listdir("packages"):
    package_name, _, ext = package.rpartition(".")
    if ext.lower() == "zip":
        target_dir = f"/tmp/{package_name}"
        sys.path.append(target_dir)
        if not os.path.exists(target_dir):
            tempdir = f"/tmp/_{package_name}"
            if os.path.exists(tempdir):
                shutil.rmtree(tempdir)
            zipfile.ZipFile(f"packages/{package}", "r").extractall(tempdir)
            os.rename(tempdir, target_dir)

print("Installs finished")
