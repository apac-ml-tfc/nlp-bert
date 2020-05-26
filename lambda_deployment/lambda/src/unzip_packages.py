"""Install locally packed Lambda dependencies on import

Some dependencies are uncompressed in a local folder, others get unzipped to /tmp. Both destinations get
appended to PATH and PYTHONPATH.

"""

# Python Built-Ins:
import datetime
import os
import shutil
import sys
import zipfile


## Step 1: Install bulky dependencies on /tmp

# The package zip contains a folder with the same name, so we just extractall to /tmp and that creates a
# subfolder:
PACKAGE_ZIP = "packages-tmpdir.zip"
TMPDIR_INSTALL_TARGET = os.path.join("/tmp", PACKAGE_ZIP.rpartition(".")[0])
INSTALL_COMPLETE_LOGFILE = os.path.join(TMPDIR_INSTALL_TARGET, "tmpinstall_complete.txt")

if (os.path.isfile(INSTALL_COMPLETE_LOGFILE)):
    print(f"Using existing {TMPDIR_INSTALL_TARGET} installation...")
else:
    print(f"Installing {PACKAGE_ZIP} to /tmp...")
    zipfile.ZipFile(PACKAGE_ZIP, "r").extractall("/tmp")
    with open(INSTALL_COMPLETE_LOGFILE, "w") as f:
        f.write(datetime.datetime.utcnow().isoformat())

# Add to PYTHONPATH so packages can be imported as normal (and PATH, just in case):
sys.path.append(TMPDIR_INSTALL_TARGET)
os.environ["PATH"] += os.pathsep + TMPDIR_INSTALL_TARGET
print(f"Installed at {TMPDIR_INSTALL_TARGET}:")
print(os.listdir(TMPDIR_INSTALL_TARGET))


## Step 2: Link other dependencies locally

# To make sure we don't fall into any relative import traps (see link below for some nice tips), we'll simply
# symlink all our packages-local objects to appear in the tmp folder alongside the extracted dependencies.
LOCAL_PACKAGE_FOLDER = os.path.join(os.path.realpath("."), "packages-local")
print(f"Linking local packages at {LOCAL_PACKAGE_FOLDER}...")
for filename in os.listdir(LOCAL_PACKAGE_FOLDER):
    os.symlink(
        os.path.join(LOCAL_PACKAGE_FOLDER, filename),
        os.path.join(TMPDIR_INSTALL_TARGET, filename),
    )

print(f"Added symlinks to local packages, now present:)
print(os.listdir(TMPDIR_INSTALL_TARGET))

print("Installs finished")
