"""Install locally packed Lambda dependencies on import

Some dependencies are uncompressed in a local folder, others get unzipped to /tmp. Both destinations get
appended to PATH and PYTHONPATH.

"""

# Python Built-Ins:
import os
import shutil
import sys
import zipfile


## Step 1: Install bulky dependencies on /tmp

# The package zip contains a folder with the same name, so we just extractall to /tmp and that creates a
# subfolder:
PACKAGE_ZIP = "packages-tmpdir.zip"
TMPDIR_INSTALL_TARGET = os.path.join("/tmp", PACKAGE_ZIP.rpartition(".")[0])
if (os.path.isdir(TMPDIR_INSTALL_TARGET)):
    print(f"Using existing {TMPDIR_INSTALL_TARGET} installation...")
else:
    print(f"Installing {PACKAGE_ZIP} to /tmp...")
    zipfile.ZipFile(PACKAGE_ZIP, "r").extractall("/tmp")

sys.path.append(TMPDIR_INSTALL_TARGET)
os.environ["PATH"] += os.pathsep + TMPDIR_INSTALL_TARGET
print(f"Installed at {TMPDIR_INSTALL_TARGET}:")
print(os.listdir(TMPDIR_INSTALL_TARGET))


## Step 2: Link other dependencies locally

cwd = os.path.realpath(".")
LOCAL_PACKAGE_FOLDER = os.path.join(cwd, "packages-local")
print(f"Linking local packages at {LOCAL_PACKAGE_FOLDER}...")
sys.path.append(LOCAL_PACKAGE_FOLDER)
os.environ["PATH"] += os.pathsep + LOCAL_PACKAGE_FOLDER
print(f"Installed at {LOCAL_PACKAGE_FOLDER}:")
print(os.listdir(LOCAL_PACKAGE_FOLDER))

print("Installs finished")
